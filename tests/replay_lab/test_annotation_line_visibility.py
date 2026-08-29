from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "human_replay_lab" / "server.py"
APP_JS = ROOT / "human_replay_lab" / "static" / "app.js"


def test_backend_has_independent_line_visibility_route():
    text = SERVER.read_text(encoding="utf-8")
    assert "annotation-line" in text
    assert "def set_annotation_line_visibility" in text
    assert '"ANNOTATION_LINE_VISIBILITY"' in text
    assert "deactivate_event" not in text[text.index("def set_annotation_line_visibility"):text.index("@staticmethod", text.index("def set_annotation_line_visibility"))]


def test_frontend_separates_line_cleanup_from_record_deletion():
    text = APP_JS.read_text(encoding="utf-8")
    assert "删线" in text
    assert "恢复线" in text
    assert "删记录" in text
    assert "/annotation-line" in text
    assert "Decision Timeline 和训练记录仍保留" in text


def test_hidden_annotations_are_skipped_only_when_drawing():
    text = APP_JS.read_text(encoding="utf-8")
    assert "annotationLineVisibilityMap" in text
    assert "isAnnotationLineVisible" in text
    assert "if (canDeleteAnnotation(ev) && !isAnnotationLineVisible(ev.id, lineVisibility)) continue;" in text
    assert "timelineEvents = state.events.filter(ev => ev.event_type !== 'ANNOTATION_LINE_VISIBILITY')" in text
