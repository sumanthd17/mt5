"""hi_en dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(hi_en): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(hi_en): BibTeX citation
_CITATION = """
"""


class HiEn(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hi_en dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(hi_en): Specifies the tfds.core.DatasetInfo object
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

    devtest = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/devtest-2021-v1.zip')

    # TODO(hi_en): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=hi/'en-hi/train.hi', target=hi/'en-hi/train.en'),
        'validation': self._generate_examples(source=devtest/'devtest/all/en-hi/dev.hi', target=devtest/'devtest/all/en-hi/dev.en'),
        'test': self._generate_examples(source=devtest/'devtest/all/en-hi/test.hi', target=devtest/'devtest/all/en-hi/test.en'),
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(hi_en): Yields (key, example) tuples from the dataset
    beam = tfds.core.lazy_imports.apache_beam

    src = tf.io.gfile.GFile(source, mode='r').readlines()
    tgt = tf.io.gfile.GFile(target, mode='r').readlines()

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

    return (beam.Create([d]) | beam.FlatMap(_process_file))
