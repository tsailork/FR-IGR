import xml.etree.ElementTree as ET
import base64
import numpy as np

tree = ET.parse('pv_outputs/plot_20.vtu')
root = tree.getroot()

for da in root.findall('.//DataArray'):
    name = da.attrib.get('Name')
    fmt = da.attrib.get('format')
    dtype_str = da.attrib.get('type')
    text = da.text.strip() if da.text else ""
    if name and fmt == 'binary' and len(text) > 0:
        raw = base64.b64decode(text)
        # First 4 bytes of VTK binary inline is header (uint32 byte count)
        num_bytes = int.from_bytes(raw[:4], byteorder='little')
        payload = raw[4:4+num_bytes]
        if dtype_str == 'Float32':
            arr = np.frombuffer(payload, dtype=np.float32)
            print(f"Array '{name}': min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6e}")

