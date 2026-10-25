# Постобработка OCR паспорта РФ

Решение тестового задания ([TASK.md](TASK.md)): по выводу OCR решить, принять документ или отклонить.

```bash
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

Работает на CPU без интернета, нужна только стандартная библиотека Python 3. При запуске
модель (словари, таблицы путаниц OCR) строится из `data/train_*.jsonl` и
`reference/subdivisions.csv`.

**Данные задания в репозиторий не входят.** Перед запуском положите файлы из задания в `data/`:

```
data/train_predictions.jsonl
data/train_labels.jsonl
data/test_predictions.jsonl
```

| Файл | Что это |
|---|---|
| `postprocess.py` | решение; параметры и пороги — в `PARAMS` в начале файла |
| `answers.jsonl` | ответы для test |
| `REPORT.md` | отчёт: что сделано, что выяснилось, прогноз для test |
| `cv.py` | 5-кратная кросс-валидация на train: `python cv.py`, сетка вариантов: `python cv.py grid` |
| `evaluate.py` | сверка с разметкой из задания |

Разбор причин отказа: `python postprocess.py data/train_predictions.jsonl train_answers.jsonl --reasons`.
