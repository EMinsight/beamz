# Interactive Jupyter Notebooks in MKDocs

This documentation site supports **interactive Jupyter notebooks** that are automatically converted and executed when the site is built. This allows for rich, interactive documentation with live code examples, plots, and widgets.

## Features

### ✅ What Works
- **Live Code Execution**: Notebooks are executed during build time
- **Interactive Plots**: Matplotlib, Plotly, and other plotting libraries
- **Interactive Widgets**: ipywidgets for interactive parameters
- **Rich Output**: Images, HTML, LaTeX, and more
- **Code Highlighting**: Syntax highlighting for all code cells
- **Dark/Light Theme**: Automatically matches the site theme

### 📝 How to Add Notebooks

1. **Create or copy** your `.ipynb` file to the `docs/` directory structure
2. **Add to navigation** in `mkdocs.yml`:
   ```yaml
   nav:
     - Examples:
       - My Interactive Tutorial: examples/my_notebook.ipynb
   ```
3. **Notebooks are automatically processed** when you run `mkdocs serve` or `mkdocs build`

### 🎛️ Interactive Features

#### Basic Plots
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

#### Interactive Widgets
```python
import ipywidgets as widgets
from IPython.display import display

@widgets.interact(x=(0, 10, 0.1))
def plot_sine(x=1.0):
    plt.plot(np.sin(x * np.linspace(0, 10, 100)))
    plt.show()
```

### 🏷️ Cell Tags for Control

You can use cell tags to control how cells are displayed:

- `hide_code` - Hides the input code, shows only output
- `hide_output` - Hides the output, shows only code
- `hide_input` - Hides the entire input cell

To add tags in Jupyter:
1. View → Cell Toolbar → Tags
2. Add tags to individual cells

### ⚙️ Configuration

The Jupyter integration is configured in `mkdocs.yml`:

```yaml
plugins:
  - mkdocs-jupyter:
      execute: true              # Execute notebooks during build
      allow_errors: false        # Stop build on errors
      theme: dark               # Match site theme
      include_source: true      # Show source code
      ignore_h1_titles: true    # Don't duplicate H1 titles
```

### 📚 Examples

Check out these interactive examples:
- [Interactive Demo](examples/simple_demo.ipynb) - Basic plots and widgets
- [MMI Tutorial](examples/MMI_powersplitter.ipynb) - Advanced BEAMZ simulation

### 🔧 Development Tips

1. **Test locally**: Always run `mkdocs serve` to test notebooks locally
2. **Keep notebooks clean**: Remove unnecessary outputs before committing
3. **Use relative imports**: Import your package modules relatively
4. **Handle dependencies**: Ensure all required packages are installed
5. **Error handling**: Set `allow_errors: false` to catch issues early

### 🚀 Advanced Features

#### Custom CSS for Notebooks
Add custom styling in `docs/stylesheets/extra.css`:

```css
/* Notebook-specific styling */
.jp-Notebook {
    /* Custom notebook styles */
}
```

#### Execution Control
You can control execution per notebook by adding metadata:

```json
{
 "metadata": {
  "mkdocs": {
   "execute": false
  }
 }
}
```

This setup gives you powerful, interactive documentation that combines the best of Jupyter notebooks with the convenience of MKDocs! 