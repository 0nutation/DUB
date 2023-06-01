# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import time

import torch
from fairseq import metrics, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
    encoders,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction
from argparse import Namespace
import json
import numpy as np

EVAL_BLEU_ORDER = 4

###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch")
class TranslationMultiSimpleEpochTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        
        #load pretrained model
        # parser.add_argument('--load-pretrained-encoder-from',default=None,
        #                     help="model to take encoder weights from (for initialization)")
        parser.add_argument('--load-pretrained-decoder-from',default=None,
                            help="model to take decoder weights from (for initialization)")             
        

        #options for evaluating BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',default=False,
                            help="if evaluate with BLEU score or not")
        parser.add_argument('--eval-bleu-args', default="{}",
                            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string')
        parser.add_argument('--eval-bleu-detok',default="space",
                            help="detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; ")
        parser.add_argument('--eval-bleu-detok_args', default="{}",
                            help="args for building the tokenizer, if needed, as JSON string")
        parser.add_argument('--eval-tokenized-bleu', action='store_true',default=False,
                            help="compute tokenized BLEU instead of sacrebleu")
        parser.add_argument('--eval-bleu-remove-bpe', default=None,
                            help="remove BPE before computing BLEU")
        parser.add_argument('--eval-bleu-print-samples', action='store_true',default=False,
                            help="print sample generations during validation")
        parser.add_argument('--eval-bleu-bpe', default=None,
                            help="use sentencepiece to bpe")
        parser.add_argument('--eval-bleu-bpe-path', default="",
                            help="path of spm.model")
        parser.add_argument('--remove-prefix-lang-tag', default = '<lang:de>',
                            help="remove language tage in hyp and ref")


        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )

        self.args.target_lang = self.target_langs[0]

    def check_dicts(self, dicts, source_langs, target_langs):
        if self.args.source_dict is not None or self.args.target_dict is not None:
            # no need to check whether the source side and target side are sharing dictionaries
            return
        src_dict = dicts[source_langs[0]]
        tgt_dict = dicts[target_langs[0]]
        for src_lang in source_langs:
            assert (
                src_dict == dicts[src_lang]
            ), "Diffrent dictionary are specified for different source languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all source languages"
        for tgt_lang in target_langs:
            assert (
                tgt_dict == dicts[tgt_lang]
            ), "Diffrent dictionary are specified for different target languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all target languages"

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
        logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info("old dataset deleted manually")
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        self.datasets[split] = self.data_manager.load_dataset(
            split,
            self.training,
            epoch=epoch,
            combine=combine,
            shard_epoch=shard_epoch,
            **kwargs,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=self.args.source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=self.args.target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset


    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        if self.args.eval_bleu:
            detok_args = json.loads(self.args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )

            if self.args.eval_bleu_bpe is None:
                self.bpe = None
            else:
                self.bpe = encoders.build_bpe(
                    Namespace(
                        bpe=self.args.eval_bleu_bpe,
                        sentencepiece_model=self.args.eval_bleu_bpe_path
                    )
                )

            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model


    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:

                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.data_manager.get_source_dictionary(self.source_langs[0])

    @property
    def target_dictionary(self):
        return self.data_manager.get_target_dictionary(self.target_langs[0])

    @property
    def prefix_lang_tag(self):
        """return prefix language tag for generate. e.g. <lang:de>"""
        return self.args.remove_prefix_lang_tag


    def create_batch_sampler_func(
        self,
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            splits = [
                s for s, _ in self.datasets.items() if self.datasets[s] == dataset
            ]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(
                f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
                logger.info(
                    f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(
                f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
            )
            logger.info(
                f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler

        return construct_batch_sampler

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}

                    #print("_bleu_sys_len",meters["_bleu_sys_len"].sum, int(meters["_bleu_sys_len"].sum))

                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            # s = self.tgt_dict.string(
            s = self.dicts[self.target_langs[0]].string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )

            if self.bpe is not None:
                s = self.bpe.decode(s)

            if self.tokenizer:
                s = self.tokenizer.decode(s)

            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(utils.strip_pad(sample["target"][i], self.dicts[self.target_langs[0]].pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            # ref = decode(utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
            #         escape_unk=True,  # don't count <unk> as matches to the hypo
            #     )
            

            #remove language tag prefix
            for tag in self.args.remove_prefix_lang_tag.split():
                hyp = hyp.replace(tag , "")
                ref = ref.replace(tag , "")
            
            if hyp == "" or hyp.split()==[]:
                hyp = '_'

            hyps.append(hyp)
            refs.append(ref)
            
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
    

    # def build_bpe(self, args):
    #     # ignore args, no one is really using it
    #     logger.info(f"tokenizer: {self.bpe}")
    #     return self.bpe