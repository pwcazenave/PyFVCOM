
def linInt(x0, x1, y0, y1, point):
    """ 
    Calculate a simple linear interpolation between two points.
    
    Specify x and y coordinates (x0, x1, y0, y1) and the point between 
    x0 and x1 at which the corresponding interpolated y value should be found.

    """

    yi = y1 + ((point - x1) * ((y1 - y0) / (x1 - x0)))

    return yi
