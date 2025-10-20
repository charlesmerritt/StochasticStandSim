from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "scripts" / "rscript" / "model_input_fert.xlsx"
YIELDS_PATH = ROOT / "scripts" / "rscript" / "yields_generated_unthinned_20241115.csv"


def _load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    """Return the shared string table for the Excel workbook."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    xml_bytes = zf.read("xl/sharedStrings.xml")
    root = ET.fromstring(xml_bytes)
    strings: List[str] = []
    for si in root.findall("main:si", ns):
        # Each shared string item can contain multiple <t> nodes (rich text).
        text = "".join(t.text or "" for t in si.findall("main:t", ns))
        strings.append(text)
    return strings


def _sheet_path(sheet_name: str) -> str:
    mapping = {
        "Read_Me": "xl/worksheets/sheet1.xml",
        "stand": "xl/worksheets/sheet2.xml",
        "regimes": "xl/worksheets/sheet3.xml",
        "specs": "xl/worksheets/sheet4.xml",
        "Sheet2": "xl/worksheets/sheet5.xml",
        "stand_alternate": "xl/worksheets/sheet6.xml",
        "thin_regimes": "xl/worksheets/sheet7.xml",
        "regime_alternate": "xl/worksheets/sheet8.xml",
    }
    try:
        return mapping[sheet_name]
    except KeyError as exc:
        raise ValueError(f"Unknown sheet {sheet_name!r}") from exc


def _read_sheet(sheet_name: str) -> List[Dict[str, Any]]:
    """Return the given Excel sheet as a list of dictionaries."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Expected Excel file at {EXCEL_PATH}")

    with zipfile.ZipFile(EXCEL_PATH) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_xml = zf.read(_sheet_path(sheet_name))

    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(sheet_xml)
    rows = root.find("main:sheetData", ns).findall("main:row", ns)

    def cell_value(cell: ET.Element) -> Any:
        value_node = cell.find("main:v", ns)
        if value_node is None:
            return None
        raw = value_node.text
        if raw is None:
            return None
        cell_type = cell.attrib.get("t")
        if cell_type == "s":
            return shared_strings[int(raw)]
        try:
            return float(raw)
        except ValueError:
            return raw

    header_cells = rows[0].findall("main:c", ns)
    headers = {}
    for cell in header_cells:
        col = "".join(filter(str.isalpha, cell.attrib["r"]))
        headers[col] = cell_value(cell)

    records: List[Dict[str, Any]] = []
    for row in rows[1:]:
        record: Dict[str, Any] = {}
        for cell in row.findall("main:c", ns):
            col = "".join(filter(str.isalpha, cell.attrib["r"]))
            header = headers.get(col)
            if header is None:
                continue
            record[header] = cell_value(cell)
        if record:
            records.append(record)
    return records


def load_stand_table() -> List[Dict[str, Any]]:
    """Return the base stand definitions from the Excel workbook."""
    return _read_sheet("stand")


def load_regime_table() -> List[Dict[str, Any]]:
    """Return the management regime definitions from the Excel workbook."""
    return _read_sheet("regimes")


def load_reference_yields(stand: str, regime: str) -> List[Dict[str, float]]:
    """Return rows from the generated yields CSV for a stand/regime pair."""
    if not YIELDS_PATH.exists():
        raise FileNotFoundError(f"Expected yield CSV at {YIELDS_PATH}")

    numeric_fields = {
        "age",
        "ba",
        "tpa",
        "hd",
        "qmd",
        "thin_remove",
        "volume",
        "saw",
        "chip",
        "pulp",
    }
    records: List[Dict[str, float]] = []
    with YIELDS_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("stand") != stand or row.get("regime") != regime:
                continue
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key in numeric_fields:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            records.append(parsed)
    if not records:
        raise ValueError(
            f"No rows found in {YIELDS_PATH.name} for stand={stand!r}, regime={regime!r}"
        )
    return records


def get_stand_record(stand_id: str) -> Mapping[str, Any]:
    """Return a single stand record by identifier."""
    matches = [row for row in load_stand_table() if row.get("stand") == stand_id]
    if not matches:
        raise ValueError(f"Stand {stand_id!r} not found in workbook")
    if len(matches) > 1:
        raise ValueError(f"Stand {stand_id!r} has multiple entries")
    return matches[0]


def get_regime_record(regime_id: str) -> Mapping[str, Any]:
    """Return a single management regime record by identifier."""
    matches = [row for row in load_regime_table() if row.get("regime") == regime_id]
    if not matches:
        raise ValueError(f"Regime {regime_id!r} not found in workbook")
    if len(matches) > 1:
        raise ValueError(f"Regime {regime_id!r} has multiple entries")
    return matches[0]
