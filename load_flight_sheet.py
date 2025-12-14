import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_flight_sheet(path: str, sheet_name: str = "flight_sheet") -> Dict[str, List[Any]]:
    """Laad het opgegeven Excel-sheet en geef een dict terug: kolomnaam -> lijst met waarden.

    - verwacht dat rij 1 kolomnamen bevat (header=0).
    - gebruikt `openpyxl` engine voor .xlsx bestanden.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")
    # zet elke kolom om naar een Python-lijst (gebruik alle rijen)
    return df.to_dict(orient="list")


if __name__ == "__main__":
    # Probeer eerst dezelfde map als dit script, anders gebruik absolute pad in workspace
    data_file = Path(__file__).parent / "data_ex2.xlsx"
    if not data_file.exists():
        data_file = Path("/Users/rosannedejong/Documents/Airline-optimalisation/Airline-optimalisation-8/data_ex2.xlsx")

    if not data_file.exists():
        print(f"Fout: kon '{data_file}' niet vinden. Plaats 'data_ex2.xlsx' in dezelfde map als dit script of pas het pad aan.")
    else:
        try:
            data = load_flight_sheet(str(data_file), sheet_name="flight_sheet")
            if not data:
                print("Sheet 'flight_sheet' is leeg of niet gevonden.")
            else:
                # bepaal aantal rijen aan de hand van de eerste kolom
                first_col = next(iter(data.values()))
                print(f"Gelaadde sheet 'flight_sheet' uit {data_file}")
                print(f"Kolommen: {len(data)}")
                print(f"Rijen: {len(first_col)}")
                print("Kolomnamen:")
                for k in data.keys():
                    print(f" - {k} (waarden: {len(data[k])})")
        except ValueError as e:
            # meestal: sheet niet gevonden — toon beschikbare sheet-namen
            try:
                xls = pd.ExcelFile(str(data_file), engine="openpyxl")
                print("Fout bij inlezen:", repr(e))
                print("Beschikbare sheets in het bestand:")
                for s in xls.sheet_names:
                    print(f" - {s}")
                print("Controleer de sheetnaam of geef de juiste naam door.")
            except Exception:
                print("Fout bij openen van Excel-bestand om sheet-namen op te vragen.")
        except Exception as e:
            print("Fout bij inlezen:", repr(e))
