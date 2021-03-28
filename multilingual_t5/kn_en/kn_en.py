"""kn_en dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(kn_en): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing wknch has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(kn_en): BibTeX citation
_CITATION = """
"""


class KnEn(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kn_en dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(kn_en): Specifies the tfds.core.DatasetInfo object
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
    # TODO(kn_en): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/ai4b-models/mT5/kn/en-kn.zip')

    # TODO(kn_en): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-kn/train/train.kn', target=path/'en-kn/train/train.en'),
        'validation': self._generate_examples(source=path/'en-kn/dev/dev.kn', target=path/'en-kn/dev/dev.en')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(kn_en): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
