# Spatial Architect Workspace (PoC)

A pure Computer Vision Proof of Concept (PoC) for a spatial software modeling environment.  
Uses hand tracking to draw nodes and edges in the air via webcam, turning your physical space into a 2D modeling canvas.

---

## Architecture

| Layer | Technology |
|---|---|
| Hand Tracking | Google MediaPipe (21 landmarks) |
| Rendering | OpenCV (direct frame overlay) |
| State Management | Python dataclasses + EMA jitter filter |
| Concurrency | Single-threaded synchronous loop (MVP) |

---

## Gestures

| Gesture | Action |
|---|---|
| Pinch in empty space | Create new Node |
| Pinch on existing Node | Start drawing Edge |
| Release Pinch on another Node | Confirm Edge |
| Release Pinch in empty space | Cancel Edge |
| `R` key | Reset canvas |
| `Q` key | Quit |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/spatial-architect-poc.git
cd spatial-architect-poc
pip install -r requirements.txt
python main.py
```

> **Python 3.10+** required.

---

## Repository Structure

```
spatial-architect-poc/
├── main.py            # Single-file application (PoC)
├── requirements.txt   # Dependencies
└── README.md
```

---

## Roadmap

- [ ] Multithreaded pipeline (Camera / Tracking / Render)
- [ ] Kalman Filter for landmark smoothing
- [ ] Node type selection (ERD / UML / Logic Gates)
- [ ] Claude API integration for code generation (SQL / C++)
- [ ] Export graph as JSON

---

## License

MIT
