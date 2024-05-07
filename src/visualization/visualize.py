from IPython.display import display_html


# Define a function to apply styles
def color_negative_red(val):
    try:
        color = "red" if val > 0 else "blue"
        return "color: %s" % color
    except TypeError:
        return "color: white"


def display_side_by_side(*args):
    html_str = ""
    for df in args:
        html_str += df.to_html()
    display_html(
        html_str.replace("table", 'table style="display:inline;margin-right: 10px;"'), raw=True
    )
