"""Raw-dump -> SLIM-KT CSV converters (interactions.csv + items.csv + folds).

Run as modules, e.g.:
    python -m slimkt.data.preprocess.eedi  --raw /path/to/eedi  --out ./slimkt_data/eedi
    python -m slimkt.data.preprocess.ednet --raw /path/to/EdNet-KT1 \
        --questions /path/to/contents/questions.csv --out ./slimkt_data/ednet
"""
