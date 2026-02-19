# Streamlit 1.54.0 — HTML/CSS Injection Notes

## ⚠️ Critical: `st.html()` renders in a sandboxed iframe

In Streamlit 1.54.0+, `st.html()` renders content inside a **sandboxed iframe**.

**This means:**
- CSS injected in the main page via `st.markdown()` does NOT apply inside `st.html()` iframes
- CSS variables (`--amber`, `--void`, etc.) are invisible to content rendered via `st.html()`
- The content inside the iframe renders without any styling = raw unstyled HTML

**DO NOT use `st.html()` for styled content in this project.**

## ✅ Correct pattern: `st.markdown(..., unsafe_allow_html=True)`

Always use this for ALL HTML rendering:

```python
st.markdown('<div class="my-class">content</div>', unsafe_allow_html=True)
st.markdown(f'<div class="panel">{variable}</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
  <p>Multi-line HTML</p>
</div>
""", unsafe_allow_html=True)
```

## ✅ CSS injection: use `components.v1.html()` with JavaScript

In Streamlit 1.54.0, `<style>` tags inside `st.markdown()` may not always apply.
The reliable approach is to inject CSS via JavaScript into the parent document:

```python
import streamlit.components.v1 as _stc

_stc.html("""
<script>
(function() {
  var style = window.parent.document.getElementById('my-app-css');
  if (!style) {
    style = window.parent.document.createElement('style');
    style.id = 'my-app-css';
    window.parent.document.head.appendChild(style);
  }
  style.textContent = `
    /* your CSS here */
    :root { --color: #fff; }
  `;
})();
</script>
""", height=0)
```

**Note:** Avoid backticks (`` ` ``) and `${...}` in CSS when embedding in JS template literals.
Escape them: `` ` `` → `\`` and `$` → `\$` if needed.

## ❌ Do not use `st.html()` for content cards/panels

```python
# WRONG — renders in iframe, CSS variables won't apply
st.html('<div class="hud-panel">...</div>')
st.html(build_candidate_card(c))

# CORRECT
st.markdown('<div class="hud-panel">...</div>', unsafe_allow_html=True)
st.markdown(build_candidate_card(c), unsafe_allow_html=True)
```

## ❌ `use_container_width` is deprecated in 1.54.0

```python
# WRONG
st.plotly_chart(fig, use_container_width=True)

# CORRECT
st.plotly_chart(fig, width="stretch")
```

## `.streamlit/config.toml`

The project uses a dark theme config at `.streamlit/config.toml`. Do not remove it — it sets the base theme that the custom CSS builds on top of.
