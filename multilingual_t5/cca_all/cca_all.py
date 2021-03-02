"""cca_all dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import random

# TODO(cca_all): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(cca_all): BibTeX citation
_CITATION = """
"""


class CcaAll(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cca_all dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cca_all): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'source': tfds.features.Text(),
            'target': tfds.features.Text(),
        }),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    hi = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/train-2021-v1-en-hi.zip')
    cca = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/hi_en/data/en-hi.zip')

    devtest = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/devtest-2021-v1.zip')

    # TODO(hi_en): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=hi/'en-hi/train.hi', target=hi/'en-hi/train.en', cca_src=cca/'en-hi/train.hi', cca_tgt=cca/'en-hi/train.en' ,mode='train'),
        'validation': self._generate_examples(source=devtest/'devtest/all/en-hi/dev.hi', target=devtest/'devtest/all/en-hi/dev.en', cca_src=cca/'cca/train.hi', cca_tgt=cca/'cca/train.en', mode='eval'),
        'test': self._generate_examples(source=devtest/'devtest/all/en-hi/test.hi', target=devtest/'devtest/all/en-hi/test.en', cca_src=cca/'cca/train.hi', cca_tgt=cca/'cca/train.en', mode='test'),
    }

  def _generate_examples(self, source, target, cca_src, cca_tgt, mode):
    """Yields examples."""
    beam = tfds.core.lazy_imports.apache_beam

    src = []
    tgt = []

    all_src = tf.io.gfile.GFile(source, mode='r').readlines()
    all_tgt = tf.io.gfile.GFile(target, mode='r').readlines()

    src.extend(all_src)
    tgt.extend(all_tgt)

    if mode == 'train':
      cca_src = tf.io.gfile.GFile(cca_src, mode='r').readlines()
      cca_tgt = tf.io.gfile.GFile(cca_tgt, mode='r').readlines()

      src.extend(cca_src)
      tgt.extend(cca_tgt)

    temp = list(zip(src, tgt))
    random.shuffle(temp)

    src, tgt = zip(*temp)

    d = {}
    d['src'] = src
    d['tgt'] = tgt

    def _process_file(d):
      src = d['src']
      tgt = d['tgt']

      for idx, row in enumerate(zip(src, tgt)):
        yield idx, {
          'source': row[0],
          'target': row[1]
        }

    return (beam.Create([d]) | beam.Map(_process_file))
