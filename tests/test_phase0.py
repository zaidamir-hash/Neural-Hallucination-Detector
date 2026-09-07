"""Phase 0 validation tests.
These are lightweight checks that do not download models or datasets.

The tests validate that the canonical configuration contains the locked Phase 0 values.
"""
import os
import re
import unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        # Fallback simple parser for this limited known structure
        cfg = {}
        # parse top-level scalar keys and simple nested keys
        current = None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^\w+:$", line):
                current = line[:-1]
                cfg[current] = {}
                continue
            if ":" in line and current:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # try basic types
                if v in ("true", "True"):
                    val = True
                elif v in ("false", "False"):
                    val = False
                else:
                    try:
                        if v.startswith('"') and v.endswith('"'):
                            val = v.strip('"')
                        else:
                            val = float(v) if '.' in v or 'e' in v.lower() else int(v)
                    except Exception:
                        val = v
                cfg[current][k] = val
        return cfg


class Phase0Tests(unittest.TestCase):
    def setUp(self):
        self.cfg_path = ROOT.joinpath("configs", "default.yaml")
        self.assertTrue(self.cfg_path.exists(), f"Expected config at {self.cfg_path}")
        self.cfg = load_yaml_config(self.cfg_path)

    def test_label_values(self):
        labels = self.cfg.get("labels")
        self.assertIsNotNone(labels, "labels section missing in config")
        self.assertEqual(labels.get("grounded"), 0, "grounded label must equal 0")
        self.assertEqual(labels.get("hallucinated"), 1, "hallucinated label must equal 1")

    def test_positive_class_documented(self):
        # Primary metric is recorded under training.primary_metric
        training = self.cfg.get("training") or {}
        primary_metric = training.get("primary_metric")
        self.assertIsNotNone(primary_metric, "primary_metric not defined in training section")
        self.assertIn("hallucinated", primary_metric)

    def test_split_fractions_and_seed(self):
        split = self.cfg.get("split")
        self.assertIsNotNone(split, "split section missing")
        self.assertAlmostEqual(float(split.get("train")), 0.70, places=6)
        self.assertAlmostEqual(float(split.get("validation")), 0.15, places=6)
        self.assertAlmostEqual(float(split.get("test")), 0.15, places=6)
        total = float(split.get("train")) + float(split.get("validation")) + float(split.get("test"))
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertIn("seed", split)

    def test_mlp_seeds_count(self):
        mlp_seeds = self.cfg.get("mlp_seeds")
        self.assertIsInstance(mlp_seeds, list)
        self.assertEqual(len(mlp_seeds), 3, "Exactly three MLP seeds must be defined")

    def test_encoder_settings(self):
        enc = self.cfg.get("encoder")
        self.assertIsNotNone(enc)
        self.assertEqual(enc.get("model"), "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(int(enc.get("embedding_dim")), 384)
        # normalized may be boolean or string in fallback parser
        self.assertTrue(str(enc.get("normalized")).lower() in ("true", "yes", "1"))

    def test_context_chunking(self):
        chunk = self.cfg.get("context_chunking")
        self.assertIsNotNone(chunk)
        self.assertEqual(int(chunk.get("max_chunk_tokens")), 192)
        self.assertEqual(int(chunk.get("chunk_overlap_tokens")), 32)

    def test_primary_feature_contract(self):
        pf = self.cfg.get("primary_features")
        self.assertIsNotNone(pf)
        self.assertEqual(int(pf.get("interaction_blocks")), 4)
        scalar = pf.get("scalar_features")
        self.assertIsInstance(scalar, list)
        self.assertEqual(len(scalar), 7)
        self.assertEqual(int(pf.get("total_dim")), 1543)

    def test_mlp_hyperparameters(self):
        mlp = self.cfg.get("mlp")
        self.assertIsNotNone(mlp)
        arch = mlp.get("architecture")
        self.assertEqual(list(map(int, arch)), [1543, 256, 64, 1])
        dropout = mlp.get("dropout")
        self.assertEqual([float(dropout[0]), float(dropout[1])], [0.20, 0.10])
        self.assertAlmostEqual(float(mlp.get("learning_rate")), 1e-3)
        self.assertAlmostEqual(float(mlp.get("weight_decay")), 1e-4)
        self.assertEqual(int(mlp.get("batch_size")), 128)
        self.assertEqual(int(mlp.get("max_epochs")), 30)
        self.assertEqual(int(mlp.get("early_stopping_patience")), 5)

    def test_config_loader_path_resolution(self):
        # Load the config.py module directly from its path and validate DEFAULT_CONFIG_PATH exists
        cfg_module_path = ROOT.joinpath("src", "hallucination_detector", "config.py")
        spec = importlib.util.spec_from_file_location("hd_config", str(cfg_module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # module should define DEFAULT_CONFIG_PATH
        self.assertTrue(hasattr(module, "DEFAULT_CONFIG_PATH"))
        default_path = module.DEFAULT_CONFIG_PATH
        self.assertTrue(Path(default_path).exists(), f"Config loader DEFAULT_CONFIG_PATH does not resolve to an existing file: {default_path}")

    def test_package_init_file(self):
        pkg = ROOT.joinpath("src", "hallucination_detector")
        init_file = pkg.joinpath("__init__.py")
        self.assertTrue(pkg.exists() and pkg.is_dir(), f"Package directory missing: {pkg}")
        self.assertTrue(init_file.exists(), "__init__.py is required in the package root")
        content = init_file.read_text(encoding="utf-8")
        self.assertIn("DEFAULT_CONFIG_PATH", content)

    def test_repo_directories_present(self):
        expected = [
            "configs",
            os.path.join("src", "hallucination_detector"),
            "data",
            "tests",
            "results",
            "notebooks",
        ]
        for path in expected:
            p = ROOT.joinpath(path)
            self.assertTrue(p.exists(), f"Expected repository folder: {p}")


if __name__ == "__main__":
    unittest.main()
