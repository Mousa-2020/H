"""
Spatial Architect Workspace — MVP (Single-File PoC)
Requirements: pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import math
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4


# ─────────────────────────────────────────────
# SECTION 1: DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Node:
    id: UUID
    x: int
    y: int
    label: str
    radius: int = 36

@dataclass
class Edge:
    id: UUID
    source_id: UUID
    target_id: UUID


# ─────────────────────────────────────────────
# SECTION 2: SPATIAL GRAPH MANAGER
# ─────────────────────────────────────────────

class SpatialGraphManager:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._node_counter: int = 0

    def add_node(self, x: int, y: int) -> Node:
        self._node_counter += 1
        node = Node(
            id=uuid4(),
            x=x,
            y=y,
            label=f"E{self._node_counter}"
        )
        self.nodes.append(node)
        return node

    def add_edge(self, source_id: UUID, target_id: UUID) -> Optional[Edge]:
        if source_id == target_id:
            return None
        for e in self.edges:
            if (e.source_id == source_id and e.target_id == target_id) or \
               (e.source_id == target_id and e.target_id == source_id):
                return None
        edge = Edge(id=uuid4(), source_id=source_id, target_id=target_id)
        self.edges.append(edge)
        return edge

    def hit_test(self, x: int, y: int) -> Optional[Node]:
        closest: Optional[Node] = None
        min_dist = float('inf')
        for node in self.nodes:
            dist = math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
            if dist <= node.radius and dist < min_dist:
                min_dist = dist
                closest = node
        return closest

    def get_node_by_id(self, node_id: UUID) -> Optional[Node]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def render(self, frame):
        for edge in self.edges:
            src = self.get_node_by_id(edge.source_id)
            tgt = self.get_node_by_id(edge.target_id)
            if src and tgt:
                cv2.line(frame, (src.x, src.y), (tgt.x, tgt.y),
                         (180, 220, 255), 2, cv2.LINE_AA)

        for node in self.nodes:
            cv2.circle(frame, (node.x, node.y), node.radius,
                       (100, 200, 255), 2, cv2.LINE_AA)
            overlay = frame.copy()
            cv2.circle(overlay, (node.x, node.y), node.radius - 2,
                       (30, 60, 100), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            label_size = cv2.getTextSize(node.label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.55, 2)[0]
            text_x = node.x - label_size[0] // 2
            text_y = node.y + label_size[1] // 2
            cv2.putText(frame, node.label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 240, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# SECTION 3: HAND TRACKER
# ─────────────────────────────────────────────

class HandTracker:
    PINCH_THRESHOLD = 0.055

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.70
        )
        self.mp_draw = mp.solutions.drawing_utils
        self._smooth_x: float = 0.0
        self._smooth_y: float = 0.0
        self._alpha: float = 0.40

    def process(self, frame) -> tuple[Optional[tuple[int, int]], bool, any]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, False, results

        h, w = frame.shape[:2]
        hand = results.multi_hand_landmarks[0]

        idx_tip   = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        raw_x = int(idx_tip.x * w)
        raw_y = int(idx_tip.y * h)

        self._smooth_x = self._alpha * raw_x + (1 - self._alpha) * self._smooth_x
        self._smooth_y = self._alpha * raw_y + (1 - self._alpha) * self._smooth_y
        finger_pos = (int(self._smooth_x), int(self._smooth_y))

        pinch_dist = math.sqrt(
            (idx_tip.x - thumb_tip.x) ** 2 +
            (idx_tip.y - thumb_tip.y) ** 2
        )
        is_pinch = pinch_dist < self.PINCH_THRESHOLD

        return finger_pos, is_pinch, results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(80, 160, 80), thickness=1, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(60, 120, 60), thickness=1)
                )


# ─────────────────────────────────────────────
# SECTION 4: INTERACTION STATE MACHINE
# ─────────────────────────────────────────────

class DrawingState:
    IDLE     = "IDLE"
    PINCHING = "PINCHING"
    COOLDOWN = "COOLDOWN"


class InteractionManager:
    COOLDOWN_FRAMES = 18

    def __init__(self, graph: SpatialGraphManager):
        self.graph = graph
        self.state: str = DrawingState.IDLE
        self.edge_source_id: Optional[UUID] = None
        self.preview_pos: Optional[tuple[int, int]] = None
        self._cooldown_counter: int = 0
        self._prev_pinch: bool = False

    def update(self, finger_pos: Optional[tuple[int, int]], is_pinch: bool):
        if finger_pos is None:
            self._prev_pinch = False
            return

        fx, fy = finger_pos
        pinch_started  = is_pinch and not self._prev_pinch
        pinch_released = not is_pinch and self._prev_pinch

        if self.state == DrawingState.COOLDOWN:
            self._cooldown_counter -= 1
            if self._cooldown_counter <= 0:
                self.state = DrawingState.IDLE
            self._prev_pinch = is_pinch
            return

        if self.state == DrawingState.IDLE:
            if pinch_started:
                hit = self.graph.hit_test(fx, fy)
                if hit:
                    self.edge_source_id = hit.id
                    self.state = DrawingState.PINCHING
                else:
                    self.graph.add_node(fx, fy)
                    self._enter_cooldown()

        elif self.state == DrawingState.PINCHING:
            self.preview_pos = (fx, fy)
            if pinch_released:
                hit = self.graph.hit_test(fx, fy)
                if hit and hit.id != self.edge_source_id:
                    self.graph.add_edge(self.edge_source_id, hit.id)
                self.edge_source_id = None
                self.preview_pos = None
                self._enter_cooldown()

        self._prev_pinch = is_pinch

    def _enter_cooldown(self):
        self.state = DrawingState.COOLDOWN
        self._cooldown_counter = self.COOLDOWN_FRAMES

    def render_preview(self, frame):
        if self.state != DrawingState.PINCHING:
            return
        if not self.edge_source_id or not self.preview_pos:
            return
        src = self.graph.get_node_by_id(self.edge_source_id)
        if src:
            cv2.line(frame, (src.x, src.y), self.preview_pos,
                     (80, 220, 140), 2, cv2.LINE_AA)
            cv2.circle(frame, self.preview_pos, 8,
                       (80, 220, 140), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# SECTION 5: HUD RENDERER
# ─────────────────────────────────────────────

def render_hud(frame, state: str, node_count: int, edge_count: int,
               is_pinch: bool, finger_pos: Optional[tuple[int, int]]):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 52), (15, 15, 25), -1)

    state_color = {
        DrawingState.IDLE:     (100, 200, 100),
        DrawingState.PINCHING: (80, 160, 255),
        DrawingState.COOLDOWN: (180, 180, 60),
    }.get(state, (200, 200, 200))

    cv2.putText(frame, f"STATE: {state}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2, cv2.LINE_AA)

    info_text = f"Nodes: {node_count}   Edges: {edge_count}"
    cv2.putText(frame, info_text, (w // 2 - 100, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

    pinch_str = "PINCH: ON" if is_pinch else "PINCH: OFF"
    pinch_col = (60, 220, 120) if is_pinch else (80, 80, 100)
    cv2.putText(frame, pinch_str, (w - 170, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_col, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 38), (w, h), (15, 15, 25), -1)
    hint = "Pinch in empty space: New Node  |  Pinch on Node: Draw Edge  |  Q: Quit  |  R: Reset"
    cv2.putText(frame, hint, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1, cv2.LINE_AA)

    if finger_pos:
        cv2.circle(frame, finger_pos, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, finger_pos, 3,
                   (60, 220, 120) if is_pinch else (200, 200, 200),
                   -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# SECTION 6: APPLICATION ENTRY POINT
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    graph       = SpatialGraphManager()
    tracker     = HandTracker()
    interaction = InteractionManager(graph)

    WINDOW_NAME = "Spatial Architect Workspace"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        finger_pos, is_pinch, results = tracker.process(frame)
        interaction.update(finger_pos, is_pinch)

        tracker.draw_landmarks(frame, results)
        graph.render(frame)
        interaction.render_preview(frame)
        render_hud(
            frame,
            state=interaction.state,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            is_pinch=is_pinch,
            finger_pos=finger_pos
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            graph.nodes.clear()
            graph.edges.clear()
            graph._node_counter = 0
            interaction.state = DrawingState.IDLE
            interaction.edge_source_id = None
            interaction.preview_pos = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

