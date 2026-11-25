---
name: read-cdf
description: >-
  Reads NASA CDF (Common Data Format) files using spacepy.pycdf in strict
  read-only mode. Use whenever reading, inspecting, or extracting data from a
  .cdf file — including variable values, per-variable metadata attributes, and
  global CDF metadata. Triggers on questions like "what variables does this
  CDF have?", "what does the CATDESC say?", "what are the global attributes?",
  or any task that opens a .cdf file with spacepy.pycdf.
disable-model-invocation: false
---

# Read CDF

Read NASA CDF files with `spacepy.pycdf` in strict read-only mode. Extract variable values, per-variable attributes, and global attributes — never mutate the file.

## When to Use

- About to open a `.cdf` file with `spacepy.pycdf` (in code, scripts, REPL, tests).
- Asked to inspect, list, or summarize what is inside a CDF (variables, dimensions, metadata).
- Asked to extract values from a specific variable, or to read a metadata attribute (variable-level or global).

**Do NOT use for:**

- Writing or modifying CDFs — this skill is read-only by design. If the task requires writing, stop and tell the user; do not silently switch modes.
- Non-CDF formats (NetCDF, HDF5, FITS) — `pycdf` does not handle those.

---

## Strict Read-Only Rule

**Always open a CDF with `readonly=True` and a `with` context manager. Never call any mutating API.**

```python
from spacepy.pycdf import CDF

with CDF(str(path), readonly=True) as cdf:
    ...
```

Wrap `path` in `str(...)` — `spacepy.pycdf.CDF` does not accept `pathlib.Path` directly.

### Forbidden operations

Never write code that does any of these — they all mutate the CDF:

| Operation | Why it's banned |
|-----------|-----------------|
| `CDF(path, readonly=False)` | Opens for write |
| `CDF(path, create=True)` | Creates a new CDF |
| `CDF(path, masterpath=...)` | Implies write/copy mode |
| `cdf.new(name, value, ...)` | Creates a new variable |
| `cdf[name] = value` | Creates or replaces a variable |
| `cdf[name][i] = value` (any slice assignment) | Mutates variable data |
| `del cdf[name]` | Deletes a variable |
| `cdf.attrs[key] = value` | Writes a global attribute |
| `cdf.attrs.new(...)` | Creates a global attribute |
| `del cdf.attrs[key]` | Deletes a global attribute |
| `var.attrs[key] = value` | Writes a variable attribute |
| `var.attrs.new(...)` | Creates a variable attribute |
| `del var.attrs[key]` | Deletes a variable attribute |

If the task genuinely requires any of these, stop and surface it to the user — do not silently open the file writable.

---

## Listing variables

`cdf` is dict-like over variable names.

```python
with CDF(str(path), readonly=True) as cdf:
    var_names = list(cdf)
    for name in cdf:
        ...
```

---

## Reading variable values

`cdf[name]` returns a `pycdf.Var` object — *not* the data. Index it to get a numpy array.

```python
with CDF(str(path), readonly=True) as cdf:
    var = cdf["epoch"]
    full_array = var[...]
    first_n = var[:10]
    one_record = var[0]
```

Useful properties on `pycdf.Var`:

| Property | Meaning |
|----------|---------|
| `var.shape` | Shape of the variable (records × dims) |
| `var.dtype` | numpy dtype |
| `var.type()` | CDF type code (e.g. `CDF_TIME_TT2000`, `CDF_REAL8`) |
| `len(var)` | Number of records |
| `var.rv()` | Record-varying flag (bool) |
| `var.dv()` | Per-dimension varying flags |

To handle fill values the same way production code does, use the helpers in `imap_l3_processing/cdf/cdf_utils.py`: `read_numeric_variable(var)` replaces `FILLVAL` with `NaN`, and `read_variable_and_mask_fill_values(var)` returns a masked array with `FILLVAL` entries masked. Both only read the variable.

Epoch variables (`CDF_EPOCH`, `CDF_EPOCH16`, `CDF_TIME_TT2000`) decode by default to numpy object arrays of Python `datetime.datetime` instances. Pass `raw=True` to the `CDF(...)` constructor (still with `readonly=True`) if raw integer/float epoch values are needed.

---

## Reading per-variable metadata (attributes)

`var.attrs` is dict-like. Each variable attribute typically has a single entry, and `var.attrs[key]` returns the value directly (not wrapped in a list).

```python
with CDF(str(path), readonly=True) as cdf:
    var = cdf["proton_density"]
    units = var.attrs["UNITS"]
    catdesc = var.attrs["CATDESC"]
    fillval = var.attrs["FILLVAL"]
    display = var.attrs.get("DISPLAY_TYPE", "")

    for k, v in var.attrs.items():
        print(f"{k} = {v!r}")
```

Common variable attributes (ISTP convention): `CATDESC`, `UNITS`, `VAR_TYPE`, `DEPEND_0`, `DEPEND_1`, `LABLAXIS`, `LABL_PTR_1`, `FILLVAL`, `VALIDMIN`, `VALIDMAX`, `FORMAT`, `DISPLAY_TYPE`, `SCALETYP`.

---

## Reading global metadata (attributes)

`cdf.attrs` is dict-like, but each global attribute can have **multiple entries**. `cdf.attrs[key]` returns a `gAttr` (list-like over entries), **not** a single value.

```python
with CDF(str(path), readonly=True) as cdf:
    parents = list(cdf.attrs["Parents"])
    first_parent = cdf.attrs["Parents"][0]
    all_entries = cdf.attrs["Parents"][...]

    for k in cdf.attrs:
        entries = list(cdf.attrs[k])
        print(f"{k}: {entries}")
```

Common global attributes (ISTP convention): `Project`, `Source_name`, `Discipline`, `Data_type`, `Descriptor`, `Data_version`, `Logical_source`, `Logical_source_description`, `Mission_group`, `Generation_date`, `Parents`, `TEXT`, `Instrument_type`.

> **Gotcha:** indexing a global attribute as if it were a scalar (`cdf.attrs['Logical_source']`) yields a `gAttr` object, not a string. Always use `[0]` for the first entry or `list(...)` / `[...]` for all entries. The safe defaulting pattern is `list(cdf.attrs.get('KEY', ['']))[0]`.

---

## Quick inspection idiom

When asked "what's in this CDF?":

```python
from spacepy.pycdf import CDF

with CDF(str(path), readonly=True) as cdf:
    print("=== Global attributes ===")
    for k in cdf.attrs:
        print(f"  {k}: {list(cdf.attrs[k])}")

    print("\n=== Variables ===")
    for name in cdf:
        var = cdf[name]
        catdesc = var.attrs.get("CATDESC", "")
        units = var.attrs.get("UNITS", "")
        print(f"  {name:<32} shape={var.shape} dtype={var.dtype} units={units!r}  '{catdesc}'")
```

This produces a complete read-only summary suitable for answering "what does this file contain?" without touching the data.
