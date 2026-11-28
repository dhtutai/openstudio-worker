import unittest
from sulphur_graph import build_graph


class SulphurGraphTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(model='sulphur3', prompt='A bird in a garden', width=1344, height=768, seed=123, num_frames=124, fps=24, steps=20)

    def test_text_and_photo_paths_use_sulphur_and_joint_audio(self):
        plain = build_graph(self.data, 'openstudio/job')
        photo = build_graph(self.data, 'openstudio/job', 'job.png')
        self.assertEqual(plain['model']['inputs']['unet_name'], 's3.safetensors')
        self.assertNotIn('first_frame', plain['condition']['inputs'])
        self.assertEqual(photo['condition']['inputs']['first_frame'], ['reference', 0])
        self.assertEqual(photo['video']['inputs']['audio'], ['audio', 0])
        self.assertEqual(photo['condition']['inputs']['length'], 124)
        self.assertEqual(photo['save']['inputs']['format'], 'mp4')

    def test_rejects_incompatible_inputs_before_submission(self):
        for patch in [dict(model='wan'), dict(width=832), dict(seed=True), dict(num_frames=81), dict(fps=16), dict(steps=8), dict(lora={}), dict(style={}), dict(negative_prompt='blur')]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build_graph({**self.data, **patch}, 'openstudio/job')


if __name__ == '__main__':
    unittest.main()
