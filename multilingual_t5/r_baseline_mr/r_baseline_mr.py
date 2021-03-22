"""baseline_mr dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(baseline_mr): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(baseline_mr): BibTeX citation
_CITATION = """
"""


class RBaselineMr(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for baseline_mr dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(baseline_mr): Specifies the tfds.core.DatasetInfo object
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
    # TODO(baseline_mr): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/baselines/mT5/baseline_mr/en-mr.zip')

    # TODO(baseline_mr): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=path/'en-mr/train/train.en', target=path/'en-mr/train/train.mr'),
        'validation': self._generate_examples(source=path/'en-mr/dev/dev.en', target=path/'en-mr/dev/dev.mr')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(baseline_mr): Yields (key, example) tuples from the dataset
    src = tf.io.gfile.GFile(source, 'r').readlines()
    tgt = tf.io.gfile.GFile(target, 'r').readlines()
    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
          'source': row[0],
          'target': row[1],
      }
