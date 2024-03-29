"""baseline_te dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(baseline_te): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(baseline_te): BibTeX citation
_CITATION = """
"""


class RBaselineTe(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for baseline_te dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(baseline_te): Specifies the tfds.core.DatasetInfo object
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
    # TODO(baseline_te): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/baselines/mT5/baseline_te/strict-en-te.zip')

    # TODO(baseline_te): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-te/train/train.en', target=path/'en-te/train/train.te'),
        'validation': self._generate_examples(source=path/'en-te/dev/dev.en', target=path/'en-te/dev/dev.te')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(baseline_te): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
