# Постобработка OCR паспорта РФ

Решение тестового задания ([TASK.md](TASK.md)): по выводу OCR решить, принять документ или отклонить.

```bash
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

Работает на CPU без интернета. **Нужен только Python 3.8 или новее**, сторонние библиотеки
не нужны (используется только стандартная библиотека), поэтому `requirements.txt` нет. Проверено на
Python 3.8, 3.10 и 3.12: `answers.jsonl` получается одинаковым. При запуске модель (словари,
таблицы путаниц OCR) строится из `data/train_*.jsonl` и `reference/subdivisions.csv`, это
занимает несколько секунд.

В коде есть экспериментальная модель слов на scikit-learn (`word_model_max_p` в `PARAMS`).
По умолчанию она выключена, и scikit-learn импортируется, только если её включить.

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
