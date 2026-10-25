# Постобработка OCR паспорта РФ

Решение тестового задания ([TASK.md](TASK.md)): по выводу OCR решить, принять документ или отклонить.

## Запуск

Нужен **Python 3.10 или новее** (проверено на 3.10 и 3.12). Работает на CPU, интернет нужен только для установки зависимостей.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
(на Linux/macOS: `source .venv/bin/activate`)
```bash
pip install -r requirements.txt
```
```bash
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

**Данные задания в репозиторий не входят.** Перед запуском положите файлы из задания в `data/`:

```
data/train_predictions.jsonl
data/train_labels.jsonl
data/test_predictions.jsonl
```

При запуске словари и таблицы путаниц OCR строятся из `data/train_*.jsonl`, это занимает несколько секунд.

## Зависимости

| Пакет | Зачем |
|---|---|
| `pydantic` | схема ответа: формат каждого из 10 полей описан в одном месте (`passport_ocr/schema.py`) |
| `mrz` | контрольные цифры MRZ по ICAO 9303. Валидатор пакета внутренний паспорт РФ не понимает (нет срока действия, цифры в именах), поэтому разбор строк свой |
| `pyspellchecker` | русский словарь: неуверенно прочитанное слово принимается, если это настоящее слово (+4 документа на CV без ошибок) |

## Структура

| Файл | Что это |
|---|---|
| `postprocess.py` | точка входа |
| `passport_ocr/config.py` | поля, дата проверки, все пороги (`PARAMS`) |
| `passport_ocr/pipeline.py` | решение accept / reject для документа, проверка действительности паспорта |
| `passport_ocr/mrz_zone.py` | разбор MRZ, транслитерация, выравнивание испорченной строки 2 |
| `passport_ocr/names.py` | ФИО: посимвольная сверка viz с MRZ |
| `passport_ocr/text_fields.py` | «Место рождения» и «Кем выдан»: словари, исправления, потерянные символы |
| `passport_ocr/viz.py` | чтение визуальной зоны и приведение к формату разметки |
| `passport_ocr/model.py` | словари и таблицы путаниц, собранные из train |
| `passport_ocr/schema.py` | pydantic-схема ответа |
| `cv.py` | 5-кратная кросс-валидация: `python cv.py`, варианты: `python cv.py grid`, `python cv.py sweep` |
| `answers.jsonl` | ответы для test |
| `REPORT.md` | отчёт: что сделано, что выяснилось, прогноз для test |
| `evaluate.py` | сверка с разметкой из задания |

Разбор причин отказа: `python postprocess.py data/train_predictions.jsonl train_answers.jsonl --reasons`.

`reference/subdivisions.csv` в решении не используется: формулировки «Кем выдан» в нём не совпадают с разметкой, подробнее в отчёте.
