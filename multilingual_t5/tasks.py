# Copyright 2021 The mT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Tasks to registry."""
import functools

from multilingual_t5 import preprocessors
from multilingual_t5 import utils
import multilingual_t5.indic_corpus.indic_corpus
import multilingual_t5.hi_en.hi_en
import multilingual_t5.baseline_bn.baseline_bn
import multilingual_t5.baseline_gu.baseline_gu
import multilingual_t5.baseline_hi.baseline_hi
import multilingual_t5.baseline_kn.baseline_kn
import multilingual_t5.baseline_ml.baseline_ml
import multilingual_t5.baseline_mr.baseline_mr
import multilingual_t5.baseline_or.baseline_or
import multilingual_t5.baseline_pa.baseline_pa
import multilingual_t5.baseline_ta.baseline_ta
import multilingual_t5.baseline_te.baseline_te

import multilingual_t5.r_baseline_bn.r_baseline_bn
import multilingual_t5.r_baseline_gu.r_baseline_gu
import multilingual_t5.r_baseline_hi.r_baseline_hi
import multilingual_t5.r_baseline_kn.r_baseline_kn
import multilingual_t5.r_baseline_ml.r_baseline_ml
import multilingual_t5.r_baseline_mr.r_baseline_mr
import multilingual_t5.r_baseline_pa.r_baseline_pa
import multilingual_t5.r_baseline_ta.r_baseline_ta
import multilingual_t5.r_baseline_te.r_baseline_te
import multilingual_t5.devanagari.devanagari

import multilingual_t5.ic_all_hi.ic_all_hi

'''
import multilingual_t5.bn_en.bn_en
import multilingual_t5.gu_en.gu_en
import multilingual_t5.hi_en.hi_en
import multilingual_t5.kn_en.kn_en
import multilingual_t5.ml_en.ml_en
import multilingual_t5.mr_en.mr_en
import multilingual_t5.pa_en.pa_en
import multilingual_t5.ta_en.ta_en
import multilingual_t5.te_en.te_en

import multilingual_t5.r_bn_en.r_bn_en
import multilingual_t5.r_gu_en.r_gu_en
import multilingual_t5.r_hi_en.r_hi_en
import multilingual_t5.r_kn_en.r_kn_en
import multilingual_t5.r_ml_en.r_ml_en
import multilingual_t5.r_mr_en.r_mr_en
import multilingual_t5.r_pa_en.r_pa_en
import multilingual_t5.r_ta_en.r_ta_en
import multilingual_t5.r_te_en.r_te_en
'''

import multilingual_t5.r_ic_all_bn.r_ic_all_bn
import multilingual_t5.r_ic_all_hi.r_ic_all_hi
import multilingual_t5.r_ic_all_ta.r_ic_all_ta

import multilingual_t5.r_indic_corp_bn.r_indic_corp_bn
import multilingual_t5.r_indic_corp_hi.r_indic_corp_hi
import multilingual_t5.r_indic_corp_ta.r_indic_corp_ta

import t5.data
from t5.evaluation import metrics
import tensorflow_datasets as tfds

# DEFAULT_SPM_PATH = "gs://pre-train/transliterated/tokenizer/indictrans-spiece.model"
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
# DEFAULT_SPM_PATH = "gs://pre-train/tokenizer/spiece.model"

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True),
}

MC4_LANGS = tfds.text.c4.MC4_LANGUAGES

# Multilingual BERT was trained on 104 languages. We include 103 of these
# languages, as tfds.wikipedia doesn't distinguish between simplified and
# traditional Chinese, and only contains "zh" (which is a mix of simplified
# and traditional).
# https://github.com/google-research/bert/blob/master/multilingual.md
WIKI_LANGS = [
    "af",
    "an",
    "ar",
    "ast",
    "az",
    "azb",
    "ba",
    "bar",
    "be",
    "bg",
    "bn",
    "bpy",
    "br",
    "bs",
    "ca",
    "ce",
    "ceb",
    "cs",
    "cv",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "fy",
    "ga",
    "gl",
    "gu",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "io",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "kn",
    "ko",
    "ky",
    "la",
    "lb",
    "lmo",
    "lt",
    "lv",
    "mg",
    "min",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "nds-nl",
    "ne",
    "new",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "pms",
    "pnb",
    "pt",
    "ro",
    "ru",
    "scn",
    "sco",
    "sh",
    "sk",
    "sl",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "vo",
    "war",
    "yo",
    "zh",
]

# =========================== Pretraining Tasks/Mixtures =======================

# mC4
for lang in MC4_LANGS:
    t5.data.TaskRegistry.add(
        "mc4.{}".format(lang.replace("-", "_")),
        t5.data.TfdsTask,
        tfds_name="c4/multilingual:3.0.1",
        splits={"train": lang, "validation": f"{lang}-validation"},
        text_preprocessor=functools.partial(
            t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
        ),
        token_preprocessor=t5.data.preprocessors.span_corruption,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[],
    )

mc4 = ["mc4.{}".format(lang.replace("-", "_")) for lang in MC4_LANGS]
t5.data.MixtureRegistry.add("mc4", mc4, default_rate=DEFAULT_MIX_RATE)

INDIC_LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "pa", "or", "ta", "te"]

for lang in INDIC_LANGS:
    t5.data.TaskRegistry.add(
        "indic_corpus.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="indic_corpus:1.0.0",
        splits={"train": lang, "validation": f"{lang}-validation"},
        text_preprocessor=functools.partial(
            t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
        ),
        token_preprocessor=t5.data.preprocessors.span_corruption,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[],
    )

indic_corpus = ["indic_corpus.{}".format(lang) for lang in INDIC_LANGS]
t5.data.MixtureRegistry.add("indic_corpus", indic_corpus, default_rate=DEFAULT_MIX_RATE)

# IndicTrans Objective
SCRIPTS = ["devanagari", "en"]

for lang in SCRIPTS:
    t5.data.TaskRegistry.add(
        "indic_trans.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="devanagari:1.0.0",
        splits={"train": lang, "validation": f"{lang}-validation"},
        text_preprocessor=functools.partial(
            t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
        ),
        token_preprocessor=t5.data.preprocessors.span_corruption,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[],
    )

indic_corpus = ["indic_trans.{}".format(lang) for lang in SCRIPTS]
t5.data.MixtureRegistry.add("indic_trans", indic_corpus, default_rate=DEFAULT_MIX_RATE)

# Wikipedia
for lang in WIKI_LANGS:
    t5.data.TaskRegistry.add(
        "wiki.{}".format(lang.replace("-", "_")),
        t5.data.TfdsTask,
        tfds_name="wikipedia/20200301.{}:1.0.0".format(lang),
        text_preprocessor=[
            functools.partial(
                t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
            ),
        ],
        token_preprocessor=t5.data.preprocessors.span_corruption,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[],
    )

wiki = ["wiki.{}".format(lang.replace("-", "_")) for lang in WIKI_LANGS]
t5.data.MixtureRegistry.add("wiki", wiki, default_rate=DEFAULT_MIX_RATE)

# Mixture of mC4 and WIKI
t5.data.MixtureRegistry.add("mc4_wiki", mc4 + wiki, default_rate=DEFAULT_MIX_RATE)

# =========================== Fine-tuning Tasks/Mixtures =======================

# ----- NMT baselines -----
# forward
t5.data.TaskRegistry.add(
    'baseline_bn',
    t5.data.TfdsTask,
    tfds_name="baseline_bn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_bn', ['baseline_bn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_gu',
    t5.data.TfdsTask,
    tfds_name="baseline_gu:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='gujarati',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_gu', ['baseline_gu'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_hi',
    t5.data.TfdsTask,
    tfds_name="baseline_hi:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_hi', ['baseline_hi'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_kn',
    t5.data.TfdsTask,
    tfds_name="baseline_kn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='kannada',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_kn', ['baseline_kn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_ml',
    t5.data.TfdsTask,
    tfds_name="baseline_ml:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='malayalam',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_ml', ['baseline_ml'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_mr',
    t5.data.TfdsTask,
    tfds_name="baseline_mr:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='marathi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_mr', ['baseline_mr'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_pa',
    t5.data.TfdsTask,
    tfds_name="baseline_pa:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='punjabi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_pa', ['baseline_pa'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_ta',
    t5.data.TfdsTask,
    tfds_name="baseline_ta:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_ta', ['baseline_ta'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'baseline_te',
    t5.data.TfdsTask,
    tfds_name="baseline_te:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='telugu',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('baseline_te', ['baseline_te'], default_rate=1.0)


# reverse
t5.data.TaskRegistry.add(
    'r_baseline_bn',
    t5.data.TfdsTask,
    tfds_name="r_baseline_bn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_bn', ['r_baseline_bn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_gu',
    t5.data.TfdsTask,
    tfds_name="r_baseline_gu:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='gujarati',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_gu', ['r_baseline_gu'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_hi',
    t5.data.TfdsTask,
    tfds_name="r_baseline_hi:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_hi', ['r_baseline_hi'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_kn',
    t5.data.TfdsTask,
    tfds_name="r_baseline_kn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='kannada',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_kn', ['r_baseline_kn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_ml',
    t5.data.TfdsTask,
    tfds_name="r_baseline_ml:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='malayalam',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_ml', ['r_baseline_ml'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_mr',
    t5.data.TfdsTask,
    tfds_name="r_baseline_mr:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='marathi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_mr', ['r_baseline_mr'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_pa',
    t5.data.TfdsTask,
    tfds_name="r_baseline_pa:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='punjabi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_pa', ['r_baseline_pa'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_ta',
    t5.data.TfdsTask,
    tfds_name="r_baseline_ta:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_ta', ['r_baseline_ta'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_baseline_te',
    t5.data.TfdsTask,
    tfds_name="r_baseline_te:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='telugu',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_baseline_te', ['r_baseline_te'], default_rate=1.0)

# ----- NMT -----

'''
# Our Contributions
t5.data.TaskRegistry.add(
    'bn_en',
    t5.data.TfdsTask,
    tfds_name="bn_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('bn_en', ['bn_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'gu_en',
    t5.data.TfdsTask,
    tfds_name="gu_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='gujarati',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('gu_en', ['gu_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'hi_en',
    t5.data.TfdsTask,
    tfds_name="hi_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('hi_en', ['hi_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'kn_en',
    t5.data.TfdsTask,
    tfds_name="kn_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='kannada',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('kn_en', ['kn_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'ml_en',
    t5.data.TfdsTask,
    tfds_name="ml_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='malayalam',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('ml_en', ['ml_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'mr_en',
    t5.data.TfdsTask,
    tfds_name="mr_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='marathi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('mr_en', ['mr_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'pa_en',
    t5.data.TfdsTask,
    tfds_name="pa_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='punjabi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('pa_en', ['pa_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'ta_en',
    t5.data.TfdsTask,
    tfds_name="ta_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('ta_en', ['ta_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'te_en',
    t5.data.TfdsTask,
    tfds_name="te_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='telugu',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('te_en', ['te_en'], default_rate=1.0)

# Our Contributions Reverse
t5.data.TaskRegistry.add(
    'r_bn_en',
    t5.data.TfdsTask,
    tfds_name="r_bn_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_bn_en', ['r_bn_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_gu_en',
    t5.data.TfdsTask,
    tfds_name="r_gu_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='gujarati',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_gu_en', ['r_gu_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_hi_en',
    t5.data.TfdsTask,
    tfds_name="r_hi_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_hi_en', ['r_hi_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_kn_en',
    t5.data.TfdsTask,
    tfds_name="r_kn_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='kannada',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_kn_en', ['r_kn_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_ml_en',
    t5.data.TfdsTask,
    tfds_name="r_ml_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='malayalam',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_ml_en', ['r_ml_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_mr_en',
    t5.data.TfdsTask,
    tfds_name="r_mr_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='marathi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_mr_en', ['r_mr_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_pa_en',
    t5.data.TfdsTask,
    tfds_name="r_pa_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='punjabi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_pa_en', ['r_pa_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_ta_en',
    t5.data.TfdsTask,
    tfds_name="r_ta_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_ta_en', ['r_ta_en'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_te_en',
    t5.data.TfdsTask,
    tfds_name="r_te_en:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='telugu',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_te_en', ['r_te_en'], default_rate=1.0)
'''

# IndicCorp Mined
t5.data.TaskRegistry.add(
    'r_ic_all_bn',
    t5.data.TfdsTask,
    tfds_name="r_ic_all_bn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_ic_all_bn', ['r_ic_all_bn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_ic_all_hi',
    t5.data.TfdsTask,
    tfds_name="r_ic_all_hi:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_ic_all_hi', ['r_ic_all_hi'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_ic_all_ta',
    t5.data.TfdsTask,
    tfds_name="r_ic_all_ta:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_ic_all_ta', ['r_ic_all_ta'], default_rate=1.0)

# Two stage training

t5.data.TaskRegistry.add(
    'r_indic_corp_bn',
    t5.data.TfdsTask,
    tfds_name="r_indic_corp_bn:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='bengali',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_indic_corp_bn', ['r_indic_corp_bn'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_indic_corp_hi',
    t5.data.TfdsTask,
    tfds_name="r_indic_corp_hi:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_indic_corp_hi', ['r_indic_corp_hi'], default_rate=1.0)

t5.data.TaskRegistry.add(
    'r_indic_corp_ta',
    t5.data.TfdsTask,
    tfds_name="r_indic_corp_ta:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='tamil',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('r_indic_corp_ta', ['r_indic_corp_ta'], default_rate=1.0)


# 
t5.data.TaskRegistry.add(
    'ic_all_hi',
    t5.data.TfdsTask,
    tfds_name="ic_all_hi:1.0.0",
    splits=['train', 'validation'],
    text_preprocessor=functools.partial(
        preprocessors.process_nmt,
        source_language='hindi',
        target_language='english'
    ),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu]
)

t5.data.MixtureRegistry.add('ic_all_hi', ['ic_all_hi'], default_rate=1.0)

# ----- XNLI -----
# XNLI zero-shot task. This fine-tunes on English MNLI training data and then
# evaluates on multilingual XNLI dev/test data.

XNLI_LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

t5.data.TaskRegistry.add(
    "xnli_train",
    t5.data.TfdsTask,
    tfds_name="multi_nli:1.1.0",
    splits=["train"],
    text_preprocessor=preprocessors.process_mnli,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy],
)
for lang in XNLI_LANGS:
    t5.data.TaskRegistry.add(
        "xnli_dev_test.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="xnli:1.1.0",
        splits=["validation", "test"],
        text_preprocessor=[
            functools.partial(preprocessors.process_xnli, target_languages=[lang])
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.accuracy],
    )
    if lang == "en":
        continue
    t5.data.TaskRegistry.add(
        "xnli_translate_train.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="xtreme_xnli:1.1.0",
        splits=["train"],
        text_preprocessor=[
            functools.partial(preprocessors.process_xnli, target_languages=[lang])
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.accuracy],
    )
t5.data.TaskRegistry.add(
    "xnli_dev_test.all_langs",
    t5.data.TfdsTask,
    tfds_name="xnli:1.1.0",
    splits=["validation", "test"],
    text_preprocessor=[
        functools.partial(preprocessors.process_xnli, target_languages=XNLI_LANGS)
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy],
)
xnli_zeroshot = ["xnli_train", "xnli_dev_test.all_langs"] + [
    "xnli_dev_test.{}".format(lang) for lang in XNLI_LANGS
]
t5.data.MixtureRegistry.add("xnli_zeroshot", xnli_zeroshot, default_rate=1.0)
xnli_translate_train = xnli_zeroshot + [
    "xnli_translate_train.{}".format(lang) for lang in XNLI_LANGS if lang != "en"
]
t5.data.MixtureRegistry.add(
    "xnli_translate_train", xnli_translate_train, default_rate=1.0
)

# ----- PAWS -----
label_names = ["different_meaning", "paraphrase"]
text_preprocessor = [
    functools.partial(
        t5.data.preprocessors.glue,
        benchmark_name="paws",
        label_names=label_names,
        feature_names=["sentence1", "sentence2"],
        id_key=None,
    )
]

postprocess_fn = functools.partial(
    t5.data.postprocessors.string_label_to_class_id, label_classes=label_names
)

t5.data.TaskRegistry.add(
    "paws",
    t5.data.TfdsTask,
    tfds_name="paws_x_wiki/en:1.0.0",
    splits=["train"],
    text_preprocessor=text_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy],
)

for lang in utils.PAWSX_LANGS:
    t5.data.TaskRegistry.add(
        "pawsx_dev_test.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
        splits=["validation", "test"],
        text_preprocessor=text_preprocessor,
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=postprocess_fn,
        metric_fns=[metrics.accuracy],
    )

    # This uses machine translations provided by the PAWS-X paper.
    t5.data.TaskRegistry.add(
        "pawsx_translate_train_original.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
        splits=["train"],
        text_preprocessor=text_preprocessor,
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=postprocess_fn,
        metric_fns=[metrics.accuracy],
    )

    if lang != "en":
        # This uses machine translations provided by the XTREME paper.
        t5.data.TaskRegistry.add(
            "pawsx_translate_train.{}".format(lang),
            t5.data.TfdsTask,
            tfds_name="xtreme_pawsx/{}:1.0.0".format(lang),
            splits=["train"],
            text_preprocessor=text_preprocessor,
            output_features=DEFAULT_OUTPUT_FEATURES,
            postprocess_fn=postprocess_fn,
            metric_fns=[metrics.accuracy],
        )

t5.data.TaskRegistry.add(
    "pawsx_dev_test.all_langs",
    t5.data.Task,
    splits=["validation", "test"],
    dataset_fn=utils.pawsx_all_langs_dataset_fn,
    text_preprocessor=text_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy],
)

# PAWSX Zero-Shot
pawsx_eval = ["pawsx_dev_test.{}".format(lang) for lang in utils.PAWSX_LANGS] + [
    "pawsx_dev_test.all_langs"
]
pawsx = ["paws"] + pawsx_eval
t5.data.MixtureRegistry.add("pawsx_zeroshot", pawsx, default_rate=1.0)

pawsx_translate_train = (
    ["paws"]
    + [
        "pawsx_translate_train.{}".format(lang)
        for lang in utils.PAWSX_LANGS
        if lang != "en"
    ]
    + pawsx_eval
)
t5.data.MixtureRegistry.add(
    "pawsx_translate_train", pawsx_translate_train, default_rate=1.0
)

pawsx_translate_train_original = [
    "pawsx_translate_train_original.{}".format(lang) for lang in utils.PAWSX_LANGS
] + pawsx_eval
t5.data.MixtureRegistry.add(
    "pawsx_translate_train_original", pawsx_translate_train, default_rate=1.0
)


# ----- TyDiQA GoldP-----
# The "validation" split contains all the validation examples for all the
# individual languages together.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]

t5.data.TaskRegistry.add(
    "tydiqa_train_dev",
    t5.data.TfdsTask,
    tfds_name="tydi_qa/goldp:2.0.0",
    splits=["train", "validation"],
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

for lang in TYDIQA_LANGS:
    t5.data.TaskRegistry.add(
        "tydiqa_dev.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="tydi_qa/goldp:2.0.0",
        splits={"validation": "validation-{}".format(lang)},
        text_preprocessor=preprocessors.xquad,
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

tydiqa = ["tydiqa_train_dev"] + ["tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS]
t5.data.MixtureRegistry.add("tydiqa", tydiqa, default_rate=1.0)

# ----- TyDiQA GoldP Zero-Shot-----
# This Zero-Shot setting matches the XTREME setup, where training is done on
# the English data of TyDiQA. In the TyDiQA paper, fine-tuning was done on
# SQuAD for zero-shot evaluation.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
t5.data.TaskRegistry.add(
    "tydiqa_train.en",
    t5.data.TfdsTask,
    tfds_name="tydi_qa/goldp:2.0.0",
    splits=["train"],
    text_preprocessor=[
        preprocessors.xquad,
        functools.partial(preprocessors.filter_tydiqa_by_language, lang="english"),
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

tydiqa_zeroshot = ["tydiqa_train.en"] + [
    "tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS
]
t5.data.MixtureRegistry.add("tydiqa_zeroshot", tydiqa_zeroshot, default_rate=1.0)


# Defining translate-train tasks.
for lang in TYDIQA_LANGS:
    # Skipping English, since translate-train is not available.
    if lang == "en":
        continue
    t5.data.TaskRegistry.add(
        "tydiqa_translate_train.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="tydi_qa/goldp:2.0.0",
        splits={"train": "translate-train-{}".format(lang)},
        text_preprocessor=preprocessors.xquad,
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

tydiqa_translate_train = (
    ["tydiqa_train.en"]
    + [f"tydiqa_translate_train.{lang}" for lang in TYDIQA_LANGS if lang != "en"]
    + [f"tydiqa_dev.{lang}" for lang in TYDIQA_LANGS]
)
t5.data.MixtureRegistry.add(
    "tydiqa_translate_train", tydiqa_translate_train, default_rate=1.0
)

# ----- English SQUAD -----
t5.data.TaskRegistry.add(
    "squad_train_dev",
    t5.data.TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    splits=["train", "validation"],
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

# ----- XQuAD -----
for lang in utils.XQUAD_LANGS_TRAIN_DEV:
    t5.data.TaskRegistry.add(
        "xquad_translate_train_dev.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="xquad/{}:2.0.0".format(lang),
        splits={"train": "translate-train", "validation": "translate-dev"},
        text_preprocessor=preprocessors.xquad,
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

for lang in utils.XQUAD_LANGS_TEST:
    t5.data.TaskRegistry.add(
        "xquad_test.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="xquad/{}:2.0.0".format(lang),
        splits=["test"],
        text_preprocessor=preprocessors.xquad,
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

# Additional test task containing all the languages.
t5.data.TaskRegistry.add(
    "xquad_test.all_langs",
    splits=["test"],
    dataset_fn=utils.xquad_all_langs_dataset_fn,
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

# XQuAD Zero-Shot (SQuAD train, SQuAD dev, XQuAD test).
xquad_test = ["xquad_test.{}".format(lang) for lang in utils.XQUAD_LANGS_TEST]
xquad_zeroshot = ["squad_train_dev", "xquad_test.all_langs"] + xquad_test
t5.data.MixtureRegistry.add("xquad_zeroshot", xquad_zeroshot, default_rate=1.0)

# XQuAD Translate-Train (English SQuAD, XQuAD translate-train,
# XQuAD translate-dev, XQuAD test)
# Note that the QA translate-train baselines from Hu et al (XTREME)
# do not include the English data. However, Fang et al (FILTER) do include
# English data.
xquad_translate_train = (
    [
        "xquad_translate_train_dev.{}".format(lang)
        for lang in utils.XQUAD_LANGS_TRAIN_DEV
    ]
    + ["squad_train_dev"]
    + ["xquad_test.all_langs"]
    + xquad_test
)
t5.data.MixtureRegistry.add(
    "xquad_translate_train", xquad_translate_train, default_rate=1.0
)


# ----- MLQA -----

MLQA_LANGS = ["ar", "de", "en", "es", "hi", "vi", "zh"]

for language in MLQA_LANGS:
    t5.data.TaskRegistry.add(
        "mlqa_dev_test.{}".format(language),
        t5.data.TfdsTask,
        tfds_name="mlqa/{}:1.0.0".format(language),
        splits=["validation", "test"],
        text_preprocessor=preprocessors.xquad,
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

# MLQA Zero-Shot
mlqa_dev_test = [f"mlqa_dev_test.{language}" for language in MLQA_LANGS]
mlqa_zeroshot = ["squad_train_dev"] + mlqa_dev_test
t5.data.MixtureRegistry.add("mlqa_zeroshot", mlqa_zeroshot, default_rate=1.0)

# MLQA Translate-Train
mlqa_translate_train = (
    ["xquad_translate_train_dev.{}".format(lang) for lang in MLQA_LANGS if lang != "en"]
    + ["squad_train_dev"]
    + mlqa_dev_test
)

t5.data.MixtureRegistry.add(
    "mlqa_translate_train", mlqa_translate_train, default_rate=1.0
)