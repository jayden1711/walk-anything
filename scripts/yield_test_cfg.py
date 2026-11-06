"""Write a tiny CPU trainer config for scripts/yield_test.sh. Usage: python scripts/yield_test_cfg.py OUT.yaml"""
import sys

import yaml

c = yaml.safe_load(open("configs/default.yaml"))
c.update(envs_per_body=2, workers=2, horizon=16, epochs=1, minibatches=2)
c["bodies"]["indices"] = [0, 1]
c["ckpt"]["every_seconds"] = 10**6
yaml.safe_dump(c, open(sys.argv[1], "w"))
