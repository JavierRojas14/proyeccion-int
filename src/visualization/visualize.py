# Define a function to apply styles
def color_negative_red(val):
    try:
        color = "red" if val > 0 else "blue"
        return "color: %s" % color
    except TypeError:
        return "color: white"
