# IIS-RDL: Relational Deep Learning врз rel-f1

Проектот проверува колку врските меѓу табелите во една релациска база носат предвидувачки
сигнал што се губи кога базата ќе се сведе на една рамна табела.

Базата се претвора во **хетероген временски граф** — секој ред е јазол, секој надворешен клуч
е типизирана врска, секоја временска ознака е време на јазолот — и врз него се учи граф
невронска мрежа, без рачно конструирани атрибути.

## Што прави проектот

1. Ја зема базата `rel-f1` од RelBench.
2. Ја претвора во хетероген временски граф преку примарни и надворешни клучеви.
3. Тренира три пристапи: табеларен (рамен), табеларен со рачни агрегати, и граф невронска мрежа.
4. Дефинира **нова задача** што RelBench не ја нуди: препорака на тим за возач за следната сезона.
5. Запишува резултати во `output/evaluation.csv` и извезува граф за визуелизација.

## Структура

```text
IIS-RDL-Project/
├── src/
├── data/
├── output/
├── notebooks/
├── report/
├── requirements-mac.txt
└── requirements-gpu.txt
```

## Инсталација

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements-mac.txt
```

## Стартување

```bash
python src/build_db.py                          # шема -> data/schema.json
python src/graph_builder.py                     # граф -> data/graph_stats.json
python src/visualize.py                         # .gexf и .html во output/
python src/baseline_tabular.py --mode flat
python src/baseline_tabular.py --mode engineered
```

GNN делот се врти на Kaggle преку `notebooks/kaggle_runner.ipynb`, бидејќи за Intel Mac
нема понови torch верзии.

## Фајлови во `src`

| Фајл | Што прави |
|---|---|
| `build_db.py` | ја чита `rel-f1` и ја запишува шемата во `data/schema.json` |
| `graph_builder.py` | ја гради релациската ентитет граф претстава, статистика во `data/graph_stats.json` |
| `visualize.py` | шема граф и computation graph околу едно предвидување |
| `baseline_tabular.py` | LightGBM baselines: `--mode flat` и `--mode engineered` |
| `model_gnn.py`, `train.py`, `features.py` | GNN моделот и тренирањето |
| `tasks/driver_constructor.py` | сопствената задача (препорака на тим) |
| `evaluate.py`, `ablations.py` | целосен прочит и аблации |

## Статистика на графот

**Релациски ентитет граф:** секој ред е јазол, секој надворешен клуч е врска, временската
колона е време на јазолот. 9 типа јазли, 74063 јазли, 26 типа врски, 169421 врска.
`drivers`, `constructors` и `circuits` немаат временска колона, па се секогаш видливи.

Границата е `<= t`: задачите го градат label-от од редови со `date > t`, па ред точно на
времето на предвидување не е дел од одговорот и смее да биде влез.

Растојание во скокови од `drivers` (ненасочено, по надворешни клучеви):

| Скока | Табели |
|---|---|
| 1 | `qualifying`, `results`, `standings` |
| 2 | `constructors`, `races` |
| 3 | `circuits`, `constructor_results`, `constructor_standings` |

## Резултати досега

Test сет, 3 seeds, mean:

| Задача | Метрика | `lgbm-flat` | `lgbm-engineered` |
|---|---|---|---|
| `driver-top3` | AUROC ↑ | 0.6318 | 0.8208 |
| `driver-dnf` | AUROC ↑ | 0.5251 | 0.7083 |
| `driver-position` | nMAE ↓ | 0.6054 | 0.5594 |

## Визуелизација на графот

`output/schema.gexf` е **шема графот** — јазлите се табели, врските се релации преку
надворешни клучеви. Се отвора во Gephi.

`output/subgraph.html` се отвора во прелистувач и прикажува **computation graph** за едно
предвидување: соседството што мрежата би го агрегирала за тој ентитет, на 2 скока. Ниту
еден јазол во него не е понов од времето на предвидување.
