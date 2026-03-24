import os
import pandas as pd
import matplotlib.pyplot as plt


MAKE_PLOT = True


PREFER_NORMALIZED_INPUT = True

ELEMENT_COLUMN_CANDIDATES = [
    "Element",
    "element",
    "Elem",
    "elem",
]

RAW_WT_COLUMN_CANDIDATES = [
    "wt%",
    "wt %",
    "Gew%",
    "Gew. %",
    "Massenanteil",
    "Wert",
    "value",
    "Massen-\nkonzentration\n/%",
    "Massenkonzentration /%",
    "Massenkonzentration / %",
    "Massen-Konzentration /%",
    "Massen-Konzentration / %",
]

NORMALIZED_WT_COLUMN_CANDIDATES = [
    "Norm. Massen-\nKonzentration\n/%",
    "Norm. Massen-Konzentration /%",
    "Norm. Massen-Konzentration / %",
    "Norm. Massenkonzentration /%",
    "Norm. Massenkonzentration / %",
]

OXYGEN_NAMES = {
    "O", "o",
    "Oxygen", "oxygen",
    "Sauerstoff", "sauerstoff"
}



def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def load_input_file(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(file_path)

    if ext == ".csv":
        attempts = [
            {"sep": ";", "decimal": ",", "encoding": "utf-8"},
            {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
            {"sep": ";", "decimal": ",", "encoding": "latin1"},
            {"sep": ",", "decimal": ".", "encoding": "utf-8"},
            {"sep": ",", "decimal": ".", "encoding": "utf-8-sig"},
            {"sep": ",", "decimal": ".", "encoding": "latin1"},
            {"sep": "\t", "decimal": ".", "encoding": "utf-8"},
            {"sep": "\t", "decimal": ".", "encoding": "latin1"},
        ]

        errors = []
        for cfg in attempts:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=cfg["sep"],
                    decimal=cfg["decimal"],
                    encoding=cfg["encoding"]
                )
                return df
            except Exception as exc:
                errors.append(f"{cfg} -> {exc}")

        raise ValueError("CSV konnte nicht gelesen werden.\n" + "\n".join(errors))

    raise ValueError("Nur CSV- und Excel-Dateien werden unterstützt.")


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Unterstützt zwei Formate:

    1. Langes Format:
       Element | wt%

    2. Breites Format:
       Spectrum | Kohlenstoff | Sauerstoff | Natrium | ...

    Rückgabe:
    - DataFrame mit Spalten: Element, wt%
    - Name der verwendeten wt%-Spalte bzw. Hinweis auf Wide-Format
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    element_col = find_column(df, ELEMENT_COLUMN_CANDIDATES)
    raw_wt_col = find_column(df, RAW_WT_COLUMN_CANDIDATES)
    norm_wt_col = find_column(df, NORMALIZED_WT_COLUMN_CANDIDATES)

    if element_col is not None:
        if PREFER_NORMALIZED_INPUT:
            wt_col = norm_wt_col if norm_wt_col is not None else raw_wt_col
        else:
            wt_col = raw_wt_col if raw_wt_col is not None else norm_wt_col

        if wt_col is None:
            raise ValueError(
                f"Keine passende wt%-Spalte gefunden. Gefundene Spalten: {list(df.columns)}"
            )

        out = df[[element_col, wt_col]].copy()
        out.columns = ["Element", "wt%"]

        out["wt%"] = (
            out["wt%"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        out["wt%"] = pd.to_numeric(out["wt%"], errors="coerce")

        out["Element"] = out["Element"].where(pd.notna(out["Element"]), None)
        out = out[out["Element"].notna()]
        out["Element"] = out["Element"].astype(str).str.strip()
        out = out.dropna(subset=["wt%"])

        bad_element_names = {"", "nan", "none", "summe", "sum", "gesamt", "total", "subtotal"}
        out = out[~out["Element"].str.lower().isin(bad_element_names)]
        out = out[~out["Element"].str.contains("unnamed", case=False, na=False)]

        return out.reset_index(drop=True), wt_col


    if len(df) == 0:
        raise ValueError("Die Datei enthält keine Daten.")


    ignore_cols = {
        "spectrum", "spec", "messpunkt", "punkt", "sample", "probe", "id"
    }

    value_columns = [col for col in df.columns if str(col).strip().lower() not in ignore_cols]

    if not value_columns:
        raise ValueError(
            f"Keine auswertbaren Element-Spalten gefunden. Gefundene Spalten: {list(df.columns)}"
        )


    first_row = df.iloc[0]

    out = pd.DataFrame({
        "Element": value_columns,
        "wt%": [first_row[col] for col in value_columns]
    })

    out["wt%"] = (
        out["wt%"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    out["wt%"] = pd.to_numeric(out["wt%"], errors="coerce")

    out["Element"] = out["Element"].astype(str).str.strip()
    out = out.dropna(subset=["wt%"])

    bad_element_names = {"", "nan", "none", "summe", "sum", "gesamt", "total", "subtotal"}
    out = out[~out["Element"].str.lower().isin(bad_element_names)]
    out = out[~out["Element"].str.contains("unnamed", case=False, na=False)]

    if out.empty:
        raise ValueError(
            f"Wide-Format erkannt, aber keine numerischen Elementwerte gefunden. "
            f"Gefundene Spalten: {list(df.columns)}"
        )

    return out.reset_index(drop=True), "Wide-format first row"


def normalize_without_oxygen(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Fachlich korrekt:
    - Falls O vorhanden: O entfernen und Rest auf 100 normieren
    - Falls O NICHT vorhanden: vorhandene Elemente direkt auf 100 normieren
    """
    df = df.copy()

    oxygen_mask = df["Element"].astype(str).str.strip().isin(OXYGEN_NAMES)

    if oxygen_mask.any():
        oxygen_value = df.loc[oxygen_mask, "wt%"].sum()
        work_df = df.loc[~oxygen_mask].copy()
        oxygen_found = True
    else:
        oxygen_value = 0.0
        work_df = df.copy()
        oxygen_found = False

    if work_df.empty:
        raise ValueError("Nach Entfernen von Sauerstoff bleiben keine Elemente übrig.")

    sum_basis = work_df["wt%"].sum()

    if sum_basis <= 0:
        raise ValueError("Die Bezugs-Summe ist <= 0. Normierung nicht möglich.")

    work_df["Original_wt%"] = work_df["wt%"]
    work_df["Normiert_ohne_O_wt%"] = (work_df["Original_wt%"] / sum_basis) * 100.0

    work_df["Original_wt%"] = work_df["Original_wt%"].round(4)
    work_df["Normiert_ohne_O_wt%"] = work_df["Normiert_ohne_O_wt%"].round(4)

    rounded_sum = round(work_df["Normiert_ohne_O_wt%"].sum(), 4)
    correction = round(100.0 - rounded_sum, 4)

    if correction != 0 and len(work_df) > 0:
        last_idx = work_df.index[-1]
        work_df.loc[last_idx, "Normiert_ohne_O_wt%"] = round(
            work_df.loc[last_idx, "Normiert_ohne_O_wt%"] + correction, 4
        )

    final_sum = round(work_df["Normiert_ohne_O_wt%"].sum(), 4)
    if final_sum != 100.0:
        raise ValueError(f"Normierte Summe ist {final_sum:.4f} statt 100.0000.")

    result = work_df[["Element", "Original_wt%", "Normiert_ohne_O_wt%"]].reset_index(drop=True)

    print("\n--- Kontrollausgabe ---")
    print(f"Sauerstoff gefunden: {'Ja' if oxygen_found else 'Nein'}")
    print(f"Entfernter Sauerstoff: {oxygen_value:.4f} wt%")
    print(f"Bezugs-Summe für Normierung: {sum_basis:.4f} wt%")
    print(f"Endsumme normiert: {result['Normiert_ohne_O_wt%'].sum():.4f} wt%")

    return result, oxygen_value, sum_basis


def save_output_csv(df: pd.DataFrame, input_path: str) -> str:
    base = os.path.splitext(input_path)[0]
    output_path = base + "_korrekt_normiert_ohne_O.csv"
    df.to_csv(output_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    return output_path


def make_plot(df: pd.DataFrame, input_path: str) -> str:
    base = os.path.splitext(input_path)[0]
    plot_path = base + "_korrekt_normiert_ohne_O.png"

    plt.figure(figsize=(10, 6))
    plt.bar(df["Element"], df["Normiert_ohne_O_wt%"])
    plt.xlabel("Element")
    plt.ylabel("Normiert ohne O (wt%)")
    plt.title("REM/EDS-Elementgehalte ohne Sauerstoff, auf 100 % normiert")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def main():
    print("REM/EDS-Auswertung: Elemente ohne Sauerstoff korrekt normieren")
    file_path = input("Pfad zur CSV- oder Excel-Datei eingeben: ").strip().strip('"')

    if not os.path.exists(file_path):
        print("Fehler: Datei nicht gefunden.")
        return

    try:
        raw_df = load_input_file(file_path)
        prepared_df, used_col = prepare_dataframe(raw_df)

        print(f"\nVerwendete wt%-Spalte: {used_col}")
        print("\nEingelesene Daten:")
        print(prepared_df)

        result_df, oxygen_value, sum_basis = normalize_without_oxygen(prepared_df)

        print("\nErgebnis:")
        print(result_df)

        output_csv = save_output_csv(result_df, file_path)
        print(f"\nCSV gespeichert unter:\n{output_csv}")

        if MAKE_PLOT:
            plot_path = make_plot(result_df, file_path)
            print(f"\nDiagramm gespeichert unter:\n{plot_path}")

    except Exception as exc:
        print(f"\nFehler: {exc}")


if __name__ == "__main__":
    main()