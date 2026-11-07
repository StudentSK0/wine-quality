"""Демо классификации Wine Quality. Обучение выполняется только в ноутбуке."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "is_red",
]
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь, %",
    "is_red": "Красное вино",
}
WINE = "#822f52"
TEAL = "#1d766d"


@st.cache_resource(show_spinner="Загружаем модель…")
def load_model(path: str, artifact_stamp: int):
    """Единый объект модели на сервер; stamp обновляет кэш при замене файла."""
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_test_data(path: str, artifact_stamp: int) -> pd.DataFrame:
    """Таблица кэшируется как данные: каждому сеансу возвращается копия."""
    return pd.read_csv(path)


@st.cache_data(show_spinner="Вычисляем вероятности на тестовой выборке…")
def test_proba(_model, X: pd.DataFrame, model_stamp: int) -> np.ndarray:
    """Модель не хэшируется; X и версия артефакта входят в ключ кэша.

    Порог не является аргументом: сдвиг слайдера не вызывает повторный
    predict_proba на тестовой выборке.
    """
    return np.asarray(_model.predict_proba(X)[:, 1], dtype=float)


@st.cache_data(show_spinner=False)
def curve_data(y: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = metrics.roc_curve(y, scores)
    precision, recall, _ = metrics.precision_recall_curve(y, scores)
    return {
        "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
        "roc_auc": float(metrics.roc_auc_score(y, scores)),
        "average_precision": float(metrics.average_precision_score(y, scores)),
    }


def validate_features(frame: pd.DataFrame, feature_names: list[str]):
    """Чистая проверка схемы/значений; пригодна и для автоматических тестов."""
    if frame.empty:
        raise ValueError("CSV-файл не содержит строк с данными.")
    if frame.columns.duplicated().any():
        raise ValueError("В CSV есть повторяющиеся имена столбцов.")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError("Не хватает обязательных столбцов: " + ", ".join(missing))
    X = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(X.to_numpy(dtype=float))
    if invalid.any():
        bad_columns = X.columns[invalid.any(axis=0)].tolist()
        bad_rows = np.flatnonzero(invalid.any(axis=1))[:5] + 2
        raise ValueError(
            "Пустые, нечисловые или бесконечные значения в столбцах: "
            + ", ".join(bad_columns)
            + ". Проверьте строки CSV: " + ", ".join(map(str, bad_rows)) + "."
        )
    if not X["is_red"].isin([0, 1]).all():
        raise ValueError("В столбце is_red допустимы только 0 (белое) и 1 (красное).")
    return X


def parse_batch_csv(payload: bytes, feature_names: list[str]):
    """Разбор файла вынесен отдельно, чтобы проверять ошибки загрузки тестами."""
    text = payload.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("CSV-файл не содержит данных.")
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t")
        header = [name.strip() for name in next(csv.reader(io.StringIO(text), dialect))]
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Не удалось распознать CSV. Используйте разделитель ',' или ';' и строку заголовков.") from exc
    if len(header) != len(set(header)):
        raise ValueError("В CSV есть повторяющиеся имена столбцов.")
    frame = pd.read_csv(io.StringIO(text), sep=dialect.delimiter)
    frame.columns = frame.columns.str.strip()
    if "target" in frame.columns:
        raise ValueError("Удалите столбец target: для предсказания нужны только признаки.")
    return frame, validate_features(frame, feature_names)


def reset_threshold(value: float):
    st.session_state["threshold"] = value


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float):
    predicted = (scores >= threshold).astype(int)
    cm = metrics.confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "predicted": predicted, "cm": cm,
        "precision": float(metrics.precision_score(y, predicted, zero_division=0)),
        "recall": float(metrics.recall_score(y, predicted, zero_division=0)),
        "f1": float(metrics.f1_score(y, predicted, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def metric_figure(y: np.ndarray, scores: np.ndarray, threshold: float):
    values = threshold_metrics(y, scores, threshold)
    curves = curve_data(y, scores)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    metrics.ConfusionMatrixDisplay(
        values["cm"], display_labels=["Остальные", "Хорошие"]
    ).plot(ax=axes[0], cmap="BuPu", colorbar=False)
    axes[0].set_xlabel("Ответ модели")
    axes[0].set_ylabel("Правильный класс")
    axes[0].set_title(f"Матрица ошибок · t = {threshold:.3f}")

    axes[1].plot(curves["fpr"], curves["tpr"], color=WINE,
                 label=f"ROC-AUC = {curves['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="#9b9499", linewidth=1)
    axes[1].scatter(values["fpr"], values["tpr"], color=TEAL, s=65,
                    zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set(xlabel="False positive rate", ylabel="Recall",
                title="ROC-кривая", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].plot(curves["recall"], curves["precision"], color=WINE,
                 label=f"Average precision = {curves['average_precision']:.3f}")
    axes[2].axhline(float(y.mean()), color="#9b9499", linestyle="--",
                    label=f"Случайный отбор: {y.mean():.1%}")
    axes[2].scatter(values["recall"], values["precision"], color=TEAL,
                    s=65, zorder=3, label=f"Порог {threshold:.3f}")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="PR-кривая",
                xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[2].legend(loc="best", fontsize=8)
    for ax in axes[1:]:
        ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Wine Quality · подборка вин", page_icon="🍷", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px;}
      h1 {letter-spacing: -.035em;}
      [data-testid="stMetricValue"] {color: #822f52;}
      [data-testid="stSidebar"] {background-color: #f7f3f4;}
    </style>
    """, unsafe_allow_html=True)
    st.title("🍷 Вино в подборку")
    st.markdown("Оценка вероятности **хорошего вина — экспертная оценка не ниже 7** — по лабораторным признакам.")

    required = {name: BASE_DIR / name for name in ("model.joblib", "config.json", "test.csv")}
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        st.error("Рядом с app.py не найдены файлы: " + ", ".join(absent)
                 + ". Сначала выполните ячейку сохранения модели в ноутбуке.")
        st.stop()
    try:
        config = json.loads(required["config.json"].read_text(encoding="utf-8"))
        default_threshold = float(config["threshold"])
        if not np.isfinite(default_threshold) or not 0 <= default_threshold <= 1:
            raise ValueError("Порог в config.json должен быть числом от 0 до 1.")
        model_stamp = required["model.joblib"].stat().st_mtime_ns
        model = load_model(str(required["model.joblib"]), model_stamp)
        test = load_test_data(str(required["test.csv"]), required["test.csv"].stat().st_mtime_ns)
        feature_names = list(getattr(model, "feature_names_in_", FEATURES))
        if set(feature_names) != set(FEATURES) or len(feature_names) != 12:
            raise ValueError("Модель должна использовать 12 признаков Wine Quality.")
        X_test = validate_features(test, feature_names)
        if "target" not in test or not test["target"].isin([0, 1]).all():
            raise ValueError("В test.csv нужен столбец target с метками 0/1.")
        y_test = test["target"].to_numpy(dtype=int)
        if np.unique(y_test).size != 2:
            raise ValueError("Для ROC- и PR-кривых в test.csv нужны оба класса.")
        scores = test_proba(model, X_test, model_stamp)
        if scores.shape != (len(test),) or not np.isfinite(scores).all():
            raise ValueError("Модель вернула некорректные вероятности.")
    except Exception as exc:
        st.error(f"Не удалось загрузить данные или модель: {exc}")
        st.stop()

    st.sidebar.header("Правило отбора")
    threshold = st.sidebar.slider(
        "Минимальная вероятность хорошего вина", 0.0, 1.0,
        value=default_threshold, step=0.001, format="%.3f", key="threshold",
        help="Один порог применяется ко всем вкладкам. Решение: вероятность ≥ порога."
    )
    st.sidebar.caption(f"Сохранённый рабочий порог: {default_threshold:.4f}")
    st.sidebar.button("Вернуть рабочий порог", on_click=reset_threshold, args=(default_threshold,), key="reset_threshold")
    st.sidebar.caption(f"Тестовая выборка: {len(test):,} вин. Хороших: {y_test.mean():.1%}.")
    st.sidebar.info("Выше порог — меньше вина в подборке и больше пропущенных хороших вин. Ниже порог — шире подборка и больше риск включить посредственное вино.")
    tab_single, tab_metrics, tab_batch, tab_coef = st.tabs([
        "Оценить вино", "Модельные метрики", "Загрузить CSV", "Что выучила модель"
    ])

    with tab_single:
        st.subheader("Лабораторный анализ одного вина")
        st.caption("Диапазоны и начальные значения взяты из test.csv: минимум, максимум и медиана.")
        with st.form("single_wine"):
            columns = st.columns(3)
            values = {}
            for index, feature in enumerate(name for name in feature_names if name != "is_red"):
                lower, upper = float(X_test[feature].min()), float(X_test[feature].max())
                middle = float(X_test[feature].median())
                digits = 5 if feature == "density" else (4 if feature == "chlorides" else 3)
                step = 0.0001 if feature in ("density", "chlorides") else 0.01
                with columns[index % 3]:
                    values[feature] = st.number_input(
                        LABELS[feature], min_value=lower, max_value=upper, value=middle,
                        step=step, format=f"%.{digits}f", key=f"single_{feature}",
                        help=feature,
                    )
            values["is_red"] = int(st.toggle("Красное вино", value=bool(X_test["is_red"].median())))
            submitted = st.form_submit_button("Оценить вино", type="primary", width="stretch", key="submit_single")
        if submitted:
            try:
                single = pd.DataFrame([values], columns=feature_names)
                probability = float(model.predict_proba(single)[0, 1])
                st.session_state["single_probability"] = probability
            except Exception as exc:
                st.error(f"Не удалось оценить вино: {exc}")
        if "single_probability" in st.session_state:
            probability = st.session_state["single_probability"]
            st.metric("Вероятность хорошего вина", f"{probability:.1%}")
            if probability >= threshold:
                st.success(f"Рекомендовать: вероятность не ниже порога {threshold:.3f}.")
            else:
                st.warning(f"Не рекомендовать при пороге {threshold:.3f}.")
            st.caption("Это оценка модели по анализу, а не индивидуальная экспертная дегустация.")

    with tab_metrics:
        st.subheader("Качество и размер подборки")
        values = threshold_metrics(y_test, scores, threshold)
        cols = st.columns(4)
        cols[0].metric("Precision", f"{values['precision']:.1%}", help="Доля хороших среди рекомендованных вин.")
        cols[1].metric("Recall", f"{values['recall']:.1%}", help="Доля найденных хороших вин среди всех хороших.")
        cols[2].metric("F1", f"{values['f1']:.3f}")
        cols[3].metric("В подборке", f"{values['predicted'].sum():,} из {len(test):,}")
        fig = metric_figure(y_test, scores, threshold)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("FP — рекомендовали посредственное вино; FN — пропустили хорошее. Если рекомендаций нет, precision и F1 показаны как 0.")
        st.info("Слайдер показывает компромисс между охватом и качеством подборки. Рабочий порог выбран на валидации в ноутбуке; тестовые метрики служат финальной оценкой, а не подбором порога.")

    with tab_batch:
        st.subheader("Оценить таблицу вин")
        st.write("Загрузите CSV с 12 признаками без столбца target. Значения is_red: 0 — белое, 1 — красное.")
        example = X_test.head(5).to_csv(index=False).encode("utf-8")
        st.download_button("Скачать пример CSV", example, "example_wines.csv", "text/csv", key="download_example")
        uploaded = st.file_uploader("CSV с лабораторными признаками", type=["csv"])
        if uploaded is not None:
            try:
                frame, features = parse_batch_csv(uploaded.getvalue(), feature_names)
                extra = [column for column in frame.columns if column not in feature_names]
                if extra:
                    st.warning("Дополнительные столбцы сохранены в результате, но не используются моделью: " + ", ".join(extra))
                outside = ((features < X_test.min()) | (features > X_test.max())).any(axis=1)
                if outside.any():
                    st.warning(f"В {outside.sum()} строках есть значения вне диапазонов тестовых данных. Такие предсказания могут быть менее надёжны.")
                probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
                result = frame.copy()
                result["probability_good"] = probability
                result["prediction"] = (probability >= threshold).astype(int)
                result["decision"] = np.where(result["prediction"] == 1, "Рекомендовать", "Не рекомендовать")
                st.caption(f"Строк: {len(result):,}; рекомендовано: {result['prediction'].sum():,}; порог: {threshold:.3f}.")
                st.dataframe(result, width="stretch", hide_index=True)
                st.download_button("Скачать предсказания", result.to_csv(index=False).encode("utf-8-sig"),
                                   "wine_predictions.csv", "text/csv", key="download_predictions")
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(f"Не удалось обработать CSV: {exc}")

    with tab_coef:
        st.subheader("Какие признаки влияют на оценку")
        estimator = model.steps[-1][1]
        if not hasattr(estimator, "coef_"):
            st.error("Сохранённая модель не содержит коэффициентов логистической регрессии.")
        else:
            coefficients = pd.Series(estimator.coef_[0], index=feature_names).sort_values()
            fig, ax = plt.subplots(figsize=(10, 5.7))
            ax.barh([LABELS[name] for name in coefficients.index], coefficients.values,
                    color=[TEAL if value > 0 else WINE for value in coefficients.values])
            ax.axvline(0, color="#777", linewidth=1)
            ax.set_xlabel("Коэффициент после StandardScaler")
            ax.grid(axis="x", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            positive = coefficients[coefficients > 0].nlargest(3).index
            negative = coefficients[coefficients < 0].nsmallest(3).index
            st.write("**Сильнее за хорошее вино:** " + ", ".join(LABELS[name] for name in positive) + ".")
            st.write("**Сильнее против:** " + ", ".join(LABELS[name] for name in negative) + ".")
            st.caption("Коэффициент показывает изменение логарифма шансов при увеличении признака на одно стандартное отклонение и фиксированных остальных признаках. Это связь в модели, а не доказательство причинности.")
            table = coefficients.rename("coefficient").to_frame()
            table["odds_multiplier"] = np.exp(np.clip(table["coefficient"], -700, 700))
            st.dataframe(table, width="stretch")
    st.divider()
    st.caption("Wine Quality · логистическая регрессия · ДЗ 3")


if __name__ == "__main__":
    main()
