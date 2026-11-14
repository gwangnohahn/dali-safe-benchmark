# Datasets

| File | Dataset | In this repository | SHA-256 |
|---|---|---|---|
| `power_plant.xlsx` | CCPP (combined-cycle power plant) | yes (CC BY 4.0, UCI Machine Learning Repository) | `dd1e4ca599fd0358c3cbd0981d4138b3edca4d6d40f1764763cc54e9fa1ba3e1` |
| `steel_fatigue.xlsx` | SFS (steel fatigue) | no — obtain from the original source (see `DATA.md`) | `ccadbbae21de433e7f5a384a1028d2a90df7100d2d3c4ac65818db4697c61930` |
| `methane_conv.xlsx` | NOMC (non-oxidative methane conversion) | no — obtain from the original source (see `DATA.md`) | `13c703162c75d90d18e17d96caae8d9b7a1502e6d626db224ba1dee42e899306` |

Place the two files you obtained in this folder under the names above and check them with

```
sha256sum steel_fatigue.xlsx methane_conv.xlsx
```

The file layout (sheet, column order, preprocessing) is the one used by the DALI repository
(https://github.com/gwangnohahn/dali-inverse-design, `DATA.md` and `code/dataset/README.md`).
The real-data blocks (`realdata`, `realdata_sens`) need all three files; the synthetic blocks need none.
