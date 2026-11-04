"""Campaign workbooks to one CSV, with rows shaped like each published layout (no openpyxl needed)."""
import datetime as dt

import calibrate_scanner as cal
import puglia_campaigns as pc

D = dt.datetime

# Headers as they appear in the published workbooks
H_2013 = ("OBJECTID", "NUMERO_ORDINE", "TECNICO", "DATA_PRELIVEO", "SPECIE", "LATITUDINE", "LONGITUDINE",
          "CULTIVAR", "RISULTATO", "COMUNE")
H_2014 = ("COMUNE", "LATITUDINE", "LONGITUDINE", "SPECIE", "DATA_CAMPIONE", "RISULTATO", "SINTOMO")
H_2023 = ("ID", "DATA_RILEVAMENTO", "TIPOLOGIA", "SINTOMO", "LATITUDINE", "LONGITUDINE", "COMUNE", "RISULTATO",
          "SPECIE", "CULTIVAR")
H_2024 = ("ID", "DATA_RILEVAMENTO", "TIPOLOGIA", "SPECIE", "CULTIVAR", "SINTOMO", "LATITUDINE", "LONGITUDINE",
          "COMUNE", "RISULTATO", "SUBSPECIE")


def test_old_layout_without_ids():
    rows = [(12533, 1646, "RNL", D(2013, 11, 25), "OLIVO", 40.5036, 16.97, "****", "POSITIVO", "PALAGIANO"),
            (12534, 1647, "RNL", D(2013, 11, 25), "OLEANDRO", 40.5, 16.9, "****", "NEGATIVO", "PALAGIANO")]
    recs = list(pc.normalize(H_2013, rows, "2013_2014"))
    assert recs[0]["ID"] == "2013_2014-12533" and recs[0]["SPECIE"] == pc.OLIVE
    assert recs[0]["RISULTATO"] == "Positivo" and recs[0]["DATA_RILEVAMENTO"] == "25/11/2013"
    assert recs[1]["SPECIE"] == "OLEANDRO" and recs[0]["SINTOMO"] == ""
    rows = [("CAMPI SALENTINA", 40.44, 17.99, "Olea Europaea (olivo)", D(2014, 11, 11), "NEGATIVO", "Assente")]
    rec = next(pc.normalize(H_2014, rows, "2014_2015"))
    assert rec["ID"] == "2014_2015-r2" and rec["SPECIE"] == pc.OLIVE and rec["SINTOMO"] == "Assente"


def test_reordered_columns_starred_ids_and_subspecies():
    rows = [("1720108*", D(2024, 5, 2), "Campione", "Olivo (Olea europaea)", "Altro", "PRESENTE", 40.6, 17.1,
             "Crispiano", "Positivo", "PAUCA"),
            ("1720109*Pianta non abbattuta; vedi DDS 158/2024", D(2024, 5, 2), "Campione", "Olivo (Olea europaea)",
             None, "Assente", 40.6, 17.1, "Crispiano", "Positivo", "PAUCA")]
    a, b = pc.normalize(H_2024, rows, "2024")
    assert a["ID"] == "1720108" and a["NOTA"] == "*" and a["SUBSPECIE"] == "pauca" and a["SINTOMO"] == "Presente"
    assert b["ID"] == "1720109" and b["NOTA"].startswith("*Pianta non abbattuta") and ";" not in b["NOTA"]
    rows = [(1585075, D(2023, 7, 28), "Campione", "Assente", 40.94, 17.27, "Monopoli", "Negativo",
             "Olivo (Olea europaea)", "Altro")]
    rec = next(pc.normalize(H_2023, rows, "2023"))
    assert rec["COMUNE"] == "Monopoli" and rec["CULTIVAR"] == "Altro" and rec["LATITUDINE"] == "40.94000000"


def test_rows_without_a_date_are_skipped():
    rows = [(1, None, "Campione", "Assente", 40.9, 17.2, "Monopoli", "Negativo", "Olivo", ""), None,
            (None,) * 10]
    out = list(pc.normalize(H_2023, rows, "2023"))
    assert out == [{"skipped": True}]


def test_write_csv_dedups_and_reads_back(tmp_path):
    first = (1, D(2023, 7, 28), "Campione", "Presente", 40.95, 17.25, "Monopoli", "Positivo",
             "Olivo (Olea europaea)", "Cima di Mola")
    again = (1, D(2023, 7, 28), "Campione", "Presente", 40.95, 17.25, "Monopoli", "Positivo",
             "Olivo (Olea europaea)", "Cima di Mola")
    other = (2, "02/02/2024", "Campione", "Assente", 40.96, 17.26, "Monopoli", "Negativo",
             "Olivo (Olea europaea)", "Cellina di Nard‗")
    reused = (2, D(2023, 9, 1), "Campione", "Assente", 40.50, 17.60, "Taranto", "Positivo",
              "Olivo (Olea europaea)", "")             # same ID as `other`, a different sample
    out = tmp_path / "all.csv"
    stats = pc.write_csv(out, [("2023", H_2023, [first, other]), ("2024", H_2023, [again, reused])])
    assert stats["2023"] == {"rows": 2, "positive": 1, "duplicates": 0, "id_reused": 0, "skipped": 0}
    assert stats["2024"] == {"rows": 1, "positive": 1, "duplicates": 1, "id_reused": 1, "skipped": 0}
    s = cal.load_samples(out)                       # UTF-8 (the cultivar is not Latin-1)
    assert list(s["id"]) == ["1", "2", "2.2"] and list(s["positive"]) == [True, False, True]
    assert list(s["season"]) == [2023, 2023, 2023] and list(s["symptomatic"]) == [True, False, False]


def test_load_samples_still_reads_the_latin1_portal_export(tmp_path):
    path = tmp_path / "portal.csv"
    path.write_bytes(("ID;DATA_RILEVAMENTO;TIPOLOGIA;SPECIE;CULTIVAR;LATITUDINE;LONGITUDINE;COMUNE;RISULTATO;SINTOMO;PROVINCIA\n"
                      "1;07/02/2020;Campione;Olivo (Olea europaea);Cellina di Nardò;40,7;17,7;Carovigno;"
                      "Negativo;Assente;Brindisi\n").encode("latin-1"))
    s = cal.load_samples(path)
    assert len(s["id"]) == 1 and s["lat"][0] == 40.7
