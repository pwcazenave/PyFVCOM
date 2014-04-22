"""Copyright 2000, 2001 William McClain

    This file is part of Astrolabe.

    Astrolabe is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Astrolabe is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astrolabe; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
    """

"""Functions which calculate the deltaT correction to convert between
dynamical and universal time.

Reference: Jean Meeus, _Astronomical Algorithms_, second edition, 1998,
Willmann-Bell, Inc.

"""
from bisect import bisect
from astronomia.calendar import cal_to_jd, jd_to_cal
from astronomia.constants import seconds_per_day
from astronomia.util import polynomial

# _tbl is a list of tuples (jd, seconds), giving deltaT values for the beginnings of
# years in a historical range. [Meeus-1998: table 10.A]

# Updated with newer (better?) values in 2011.
# the following are not ALL from Meeus, but 1657-2010 are taken from
# http://www.iers.org/iers/earth/rotation/ut1lod/table2.html # NO LONGER THERE
# Found info here...
# http://maia.usno.navy.mil/ser7/historic_deltat.data
# With the main site here...
# http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

_tbl = [
     (cal_to_jd(1620), 121.0),
     (cal_to_jd(1622), 112.0),
     (cal_to_jd(1624), 103.0),
     (cal_to_jd(1626), 95.0),
     (cal_to_jd(1628), 88.0),

     (cal_to_jd(1630), 82.0),
     (cal_to_jd(1632), 77.0),
     (cal_to_jd(1634), 72.0),
     (cal_to_jd(1636), 68.0),
     (cal_to_jd(1638), 63.0),

     (cal_to_jd(1640), 60.0),
     (cal_to_jd(1642), 56.0),
     (cal_to_jd(1644), 53.0),
     (cal_to_jd(1646), 51.0),
     (cal_to_jd(1648), 48.0),

     (cal_to_jd(1650), 46.0),
     (cal_to_jd(1652), 44.0),
     (cal_to_jd(1654), 42.0),
     (cal_to_jd(1656), 40.0),

# Originally from Meeus
#    (cal_to_jd(1658), 38.0),
#
#    (cal_to_jd(1660), 35.0),
#    (cal_to_jd(1662), 33.0),
#    (cal_to_jd(1664), 31.0),
#    (cal_to_jd(1666), 29.0),
#    (cal_to_jd(1668), 26.0),
#
#    (cal_to_jd(1670), 24.0),
#    (cal_to_jd(1672), 22.0),
#    (cal_to_jd(1674), 20.0),
#    (cal_to_jd(1676), 28.0),
#    (cal_to_jd(1678), 16.0),
#
#    (cal_to_jd(1680), 14.0),
#    (cal_to_jd(1682), 12.0),
#    (cal_to_jd(1684), 11.0),
#    (cal_to_jd(1686), 10.0),
#    (cal_to_jd(1688), 9.0),
#
#    (cal_to_jd(1690), 8.0),
#    (cal_to_jd(1692), 7.0),
#    (cal_to_jd(1694), 7.0),
#    (cal_to_jd(1696), 7.0),
#    (cal_to_jd(1698), 7.0),
#
#    (cal_to_jd(1700), 7.0),
#    (cal_to_jd(1702), 7.0),
#    (cal_to_jd(1704), 8.0),
#    (cal_to_jd(1706), 8.0),
#    (cal_to_jd(1708), 9.0),
#
#    (cal_to_jd(1710), 9.0),
#    (cal_to_jd(1712), 9.0),
#    (cal_to_jd(1714), 9.0),
#    (cal_to_jd(1716), 9.0),
#    (cal_to_jd(1718), 10.0),
#
#    (cal_to_jd(1720), 10.0),
#    (cal_to_jd(1722), 10.0),
#    (cal_to_jd(1724), 10.0),
#    (cal_to_jd(1726), 10.0),
#    (cal_to_jd(1728), 10.0),
#
#    (cal_to_jd(1730), 10.0),
#    (cal_to_jd(1732), 10.0),
#    (cal_to_jd(1734), 11.0),
#    (cal_to_jd(1736), 11.0),
#    (cal_to_jd(1738), 11.0),
#
#    (cal_to_jd(1740), 11.0),
#    (cal_to_jd(1742), 11.0),
#    (cal_to_jd(1744), 12.0),
#    (cal_to_jd(1746), 12.0),
#    (cal_to_jd(1748), 12.0),
#
#    (cal_to_jd(1750), 12.0),
#    (cal_to_jd(1752), 13.0),
#    (cal_to_jd(1754), 13.0),
#    (cal_to_jd(1756), 13.0),
#    (cal_to_jd(1758), 14.0),
#
#    (cal_to_jd(1760), 14.0),
#    (cal_to_jd(1762), 14.0),
#    (cal_to_jd(1764), 14.0),
#    (cal_to_jd(1766), 15.0),
#    (cal_to_jd(1768), 15.0),
#
#    (cal_to_jd(1770), 15.0),
#    (cal_to_jd(1772), 15.0),
#    (cal_to_jd(1774), 15.0),
#    (cal_to_jd(1776), 16.0),
#    (cal_to_jd(1778), 16.0),
#
#    (cal_to_jd(1780), 16.0),
#    (cal_to_jd(1782), 16.0),
#    (cal_to_jd(1784), 16.0),
#    (cal_to_jd(1786), 16.0),
#    (cal_to_jd(1788), 16.0),
#
#    (cal_to_jd(1790), 16.0),
#    (cal_to_jd(1792), 15.0),
#    (cal_to_jd(1794), 15.0),
#    (cal_to_jd(1796), 14.0),
#    (cal_to_jd(1798), 13.0),
#
#    (cal_to_jd(1800), 13.1),
#    (cal_to_jd(1802), 12.5),
#    (cal_to_jd(1804), 12.2),
#    (cal_to_jd(1806), 12.0),
#    (cal_to_jd(1808), 12.0),
#
#    (cal_to_jd(1810), 12.0),
#    (cal_to_jd(1812), 12.0),
#    (cal_to_jd(1814), 12.0),
#    (cal_to_jd(1816), 12.0),
#    (cal_to_jd(1818), 11.9),
#
#    (cal_to_jd(1820), 11.6),
#    (cal_to_jd(1822), 11.0),
#    (cal_to_jd(1824), 10.2),
#    (cal_to_jd(1826), 9.2),
#    (cal_to_jd(1828), 8.2),
#
#    (cal_to_jd(1830), 7.1),
#    (cal_to_jd(1832), 6.2),
#    (cal_to_jd(1834), 5.6),
#    (cal_to_jd(1836), 5.4),
#    (cal_to_jd(1838), 5.3),
#
#    (cal_to_jd(1840), 5.4),
#    (cal_to_jd(1842), 5.6),
#    (cal_to_jd(1844), 5.9),
#    (cal_to_jd(1846), 6.2),
#    (cal_to_jd(1848), 6.5),
#
#    (cal_to_jd(1850), 6.8),
#    (cal_to_jd(1852), 7.1),
#    (cal_to_jd(1854), 7.3),
#    (cal_to_jd(1856), 7.5),
#    (cal_to_jd(1858), 7.6),
#
#    (cal_to_jd(1860), 7.7),
#    (cal_to_jd(1862), 7.3),
#    (cal_to_jd(1864), 6.2),
#    (cal_to_jd(1866), 5.2),
#    (cal_to_jd(1868), 2.7),
#
#    (cal_to_jd(1870), 1.4),
#    (cal_to_jd(1872), -1.2),
#    (cal_to_jd(1874), -2.8),
#    (cal_to_jd(1876), -3.8),
#    (cal_to_jd(1878), -4.8),
#
#    (cal_to_jd(1880), -5.5),
#    (cal_to_jd(1882), -5.3),
#    (cal_to_jd(1884), -5.6),
#    (cal_to_jd(1886), -5.7),
#    (cal_to_jd(1888), -5.9),
#
#    (cal_to_jd(1890), -6.0),
#    (cal_to_jd(1892), -6.3),
#    (cal_to_jd(1894), -6.5),
#    (cal_to_jd(1896), -6.2),
#    (cal_to_jd(1898), -4.7),
#
#    (cal_to_jd(1900), -2.8),
#    (cal_to_jd(1902), -0.1),
#    (cal_to_jd(1904), 2.6),
#    (cal_to_jd(1906), 5.3),
#    (cal_to_jd(1908), 7.7),
#
#    (cal_to_jd(1910), 10.4),
#    (cal_to_jd(1912), 13.3),
#    (cal_to_jd(1914), 16.0),
#    (cal_to_jd(1916), 18.2),
#    (cal_to_jd(1918), 20.2),
#
#    (cal_to_jd(1920), 21.1),
#    (cal_to_jd(1922), 22.4),
#    (cal_to_jd(1924), 23.5),
#    (cal_to_jd(1926), 23.8),
#    (cal_to_jd(1928), 24.3),
#
#    (cal_to_jd(1930), 24.0),
#    (cal_to_jd(1932), 23.9),
#    (cal_to_jd(1934), 23.9),
#    (cal_to_jd(1936), 23.7),
#    (cal_to_jd(1938), 24.0),
#
#    (cal_to_jd(1940), 24.3),
#    (cal_to_jd(1942), 25.3),
#    (cal_to_jd(1944), 26.2),
#    (cal_to_jd(1946), 27.3),
#    (cal_to_jd(1948), 28.2),
#
#    (cal_to_jd(1950), 29.1),
#    (cal_to_jd(1952), 30.0),
#    (cal_to_jd(1954), 30.7),
#    (cal_to_jd(1956), 31.4),
#    (cal_to_jd(1958), 32.2),
#
#    (cal_to_jd(1960), 33.1),
#    (cal_to_jd(1962), 34.0),
#    (cal_to_jd(1964), 35.0),
#    (cal_to_jd(1966), 36.5),
#    (cal_to_jd(1968), 38.3),
#
#    (cal_to_jd(1970), 40.2),
#    (cal_to_jd(1972), 42.2),
#    (cal_to_jd(1974), 44.5),
#    (cal_to_jd(1976), 46.5),
#    (cal_to_jd(1978), 48.5),
#
#    (cal_to_jd(1980), 50.5),
#    (cal_to_jd(1982), 52.2),
#    (cal_to_jd(1984), 53.8),
#    (cal_to_jd(1986), 54.9),
#    (cal_to_jd(1988), 55.8),
#
#    (cal_to_jd(1990), 56.9),
#    (cal_to_jd(1992), 58.3),
#    (cal_to_jd(1994), 60.0),
#    (cal_to_jd(1996), 61.6),
#    (cal_to_jd(1998), 63.0),
#
    (cal_to_jd(1657),      44     ), 
    (cal_to_jd(1658),      43     ), 
    (cal_to_jd(1659),      40     ), 
    (cal_to_jd(1660),      38     ), 
    (cal_to_jd(1661),      37     ), 
    (cal_to_jd(1662),      36     ), 
    (cal_to_jd(1663),      37     ), 
    (cal_to_jd(1664),      38     ), 
    (cal_to_jd(1665),      36     ), 
    (cal_to_jd(1666),      35     ), 
    (cal_to_jd(1667),      34     ), 
    (cal_to_jd(1668),      33     ), 
    (cal_to_jd(1669),      32     ), 
    (cal_to_jd(1670),      31     ), 
    (cal_to_jd(1671),      30     ), 
    (cal_to_jd(1672),      29     ), 
    (cal_to_jd(1673),      29     ), 
    (cal_to_jd(1674),      28     ), 
    (cal_to_jd(1675),      27     ), 
    (cal_to_jd(1676),      26     ), 
    (cal_to_jd(1677),      25     ), 
    (cal_to_jd(1678),      25     ), 
    (cal_to_jd(1679),      26     ), 
    (cal_to_jd(1680),      26     ), 
    (cal_to_jd(1681),      25     ), 
    (cal_to_jd(1682),      24     ), 
    (cal_to_jd(1683),      24     ), 
    (cal_to_jd(1684),      24     ), 
    (cal_to_jd(1685),      24     ), 
    (cal_to_jd(1686),      24     ), 
    (cal_to_jd(1687),      23     ), 
    (cal_to_jd(1688),      23     ), 
    (cal_to_jd(1689),      22     ), 
    (cal_to_jd(1690),      22     ), 
    (cal_to_jd(1691),      22     ), 
    (cal_to_jd(1692),      21     ), 
    (cal_to_jd(1693),      21     ), 
    (cal_to_jd(1694),      21     ), 
    (cal_to_jd(1695),      21     ), 
    (cal_to_jd(1696),      20     ), 
    (cal_to_jd(1697),      20     ), 
    (cal_to_jd(1698),      20     ), 
    (cal_to_jd(1699),      20     ), 
    (cal_to_jd(1700),      21     ), 
    (cal_to_jd(1701),      21     ), 
    (cal_to_jd(1702),      20     ), 
    (cal_to_jd(1703),      20     ), 
    (cal_to_jd(1704),      19     ), 
    (cal_to_jd(1705),      19     ), 
    (cal_to_jd(1706),      19     ), 
    (cal_to_jd(1707),      20     ), 
    (cal_to_jd(1708),      20     ), 
    (cal_to_jd(1709),      20     ), 
    (cal_to_jd(1710),      20     ), 
    (cal_to_jd(1711),      20     ), 
    (cal_to_jd(1712),      21     ), 
    (cal_to_jd(1713),      21     ), 
    (cal_to_jd(1714),      21     ), 
    (cal_to_jd(1715),      21     ), 
    (cal_to_jd(1716),      21     ), 
    (cal_to_jd(1717),      21     ), 
    (cal_to_jd(1718),      21     ), 
    (cal_to_jd(1719),      21     ), 
    (cal_to_jd(1720),      21.1   ), 
    (cal_to_jd(1721),      21.0   ), 
    (cal_to_jd(1722),      20.9   ), 
    (cal_to_jd(1723),      20.7   ), 
    (cal_to_jd(1724),      20.4   ), 
    (cal_to_jd(1725),      20.0   ), 
    (cal_to_jd(1726),      19.4   ), 
    (cal_to_jd(1727),      18.7   ), 
    (cal_to_jd(1728),      17.8   ), 
    (cal_to_jd(1729),      17.0   ), 
    (cal_to_jd(1730),      16.6   ), 
    (cal_to_jd(1731),      16.1   ), 
    (cal_to_jd(1732),      15.7   ), 
    (cal_to_jd(1733),      15.3   ), 
    (cal_to_jd(1734),      14.7   ), 
    (cal_to_jd(1735),      14.3   ), 
    (cal_to_jd(1736),      14.1   ), 
    (cal_to_jd(1737),      14.1   ), 
    (cal_to_jd(1738),      13.7   ), 
    (cal_to_jd(1739),      13.5   ), 
    (cal_to_jd(1740),      13.5   ), 
    (cal_to_jd(1741),      13.4   ), 
    (cal_to_jd(1742),      13.4   ), 
    (cal_to_jd(1743),      13.3   ), 
    (cal_to_jd(1744),      13.2   ), 
    (cal_to_jd(1745),      13.2   ), 
    (cal_to_jd(1746),      13.1   ), 
    (cal_to_jd(1747),      13.0   ), 
    (cal_to_jd(1748),      13.3   ), 
    (cal_to_jd(1749),      13.5   ), 
    (cal_to_jd(1750),      13.7   ), 
    (cal_to_jd(1751),      13.9   ), 
    (cal_to_jd(1752),      14.0   ), 
    (cal_to_jd(1753),      14.1   ), 
    (cal_to_jd(1754),      14.1   ), 
    (cal_to_jd(1755),      14.3   ), 
    (cal_to_jd(1756),      14.4   ), 
    (cal_to_jd(1757),      14.6   ), 
    (cal_to_jd(1758),      14.7   ), 
    (cal_to_jd(1759),      14.7   ), 
    (cal_to_jd(1760),      14.8   ), 
    (cal_to_jd(1761),      14.9   ), 
    (cal_to_jd(1762),      15.0   ), 
    (cal_to_jd(1763),      15.2   ), 
    (cal_to_jd(1764),      15.4   ), 
    (cal_to_jd(1765),      15.6   ), 
    (cal_to_jd(1766),      15.6   ), 
    (cal_to_jd(1767),      15.9   ), 
    (cal_to_jd(1768),      15.9   ), 
    (cal_to_jd(1769),      15.7   ), 
    (cal_to_jd(1770),      15.7   ), 
    (cal_to_jd(1771),      15.7   ), 
    (cal_to_jd(1772),      15.9   ), 
    (cal_to_jd(1773),      16.1   ), 
    (cal_to_jd(1774),      15.9   ), 
    (cal_to_jd(1775),      15.7   ), 
    (cal_to_jd(1776),      15.3   ), 
    (cal_to_jd(1777),      15.5   ), 
    (cal_to_jd(1778),      15.6   ), 
    (cal_to_jd(1779),      15.6   ), 
    (cal_to_jd(1780),      15.6   ), 
    (cal_to_jd(1781),      15.5   ), 
    (cal_to_jd(1782),      15.4   ), 
    (cal_to_jd(1783),      15.2   ), 
    (cal_to_jd(1784),      14.9   ), 
    (cal_to_jd(1785),      14.6   ), 
    (cal_to_jd(1786),      14.3   ), 
    (cal_to_jd(1787),      14.1   ), 
    (cal_to_jd(1788),      14.2   ), 
    (cal_to_jd(1789),      13.7   ), 
    (cal_to_jd(1790),      13.3   ), 
    (cal_to_jd(1791),      13.0   ), 
    (cal_to_jd(1792),      13.2   ), 
    (cal_to_jd(1793),      13.1   ), 
    (cal_to_jd(1794),      13.3   ), 
    (cal_to_jd(1795),      13.5   ), 
    (cal_to_jd(1796),      13.2   ), 
    (cal_to_jd(1797),      13.1   ), 
    (cal_to_jd(1798),      13.0   ), 
    (cal_to_jd(1799),      12.6   ), 
    (cal_to_jd(1800),      12.6   ), 
    (cal_to_jd(1801),      12.0   ), 
    (cal_to_jd(1802),      11.8   ), 
    (cal_to_jd(1803),      11.4   ), 
    (cal_to_jd(1804),      11.1   ), 
    (cal_to_jd(1805),      11.1   ), 
    (cal_to_jd(1806),      11.1   ), 
    (cal_to_jd(1807),      11.1   ), 
    (cal_to_jd(1808),      11.2   ), 
    (cal_to_jd(1809),      11.5   ), 
    (cal_to_jd(1810),      11.2   ), 
    (cal_to_jd(1811),      11.7   ), 
    (cal_to_jd(1812),      11.9   ), 
    (cal_to_jd(1813),      11.8   ), 
    (cal_to_jd(1814),      11.8   ), 
    (cal_to_jd(1815),      11.8   ), 
    (cal_to_jd(1816),      11.6   ), 
    (cal_to_jd(1817),      11.5   ), 
    (cal_to_jd(1818),      11.4   ), 
    (cal_to_jd(1819),      11.3   ), 
    (cal_to_jd(1820),      11.13  ), 
    (cal_to_jd(1821),      10.94  ), 
    (cal_to_jd(1822),      10.29  ), 
    (cal_to_jd(1823),       9.94  ), 
    (cal_to_jd(1824),       9.88  ), 
    (cal_to_jd(1825),       9.72  ), 
    (cal_to_jd(1826),       9.66  ), 
    (cal_to_jd(1827),       9.51  ), 
    (cal_to_jd(1828),       9.21  ), 
    (cal_to_jd(1829),       8.60  ), 
    (cal_to_jd(1830),       7.95  ), 
    (cal_to_jd(1831),       7.59  ), 
    (cal_to_jd(1832),       7.36  ), 
    (cal_to_jd(1833),       7.10  ), 
    (cal_to_jd(1834),       6.89  ), 
    (cal_to_jd(1835),       6.73  ), 
    (cal_to_jd(1836),       6.39  ), 
    (cal_to_jd(1837),       6.25  ), 
    (cal_to_jd(1838),       6.25  ), 
    (cal_to_jd(1839),       6.22  ), 
    (cal_to_jd(1840),       6.22  ), 
    (cal_to_jd(1841),       6.30  ), 
    (cal_to_jd(1842),       6.35  ), 
    (cal_to_jd(1843),       6.32  ), 
    (cal_to_jd(1844),       6.33  ), 
    (cal_to_jd(1845),       6.37  ), 
    (cal_to_jd(1846),       6.40  ), 
    (cal_to_jd(1847),       6.46  ), 
    (cal_to_jd(1848),       6.48  ), 
    (cal_to_jd(1849),       6.53  ), 
    (cal_to_jd(1850),       6.55  ), 
    (cal_to_jd(1851),       6.69  ), 
    (cal_to_jd(1852),       6.84  ), 
    (cal_to_jd(1853),       7.03  ), 
    (cal_to_jd(1854),       7.15  ), 
    (cal_to_jd(1855),       7.26  ), 
    (cal_to_jd(1856),       7.23  ), 
    (cal_to_jd(1857),       7.21  ), 
    (cal_to_jd(1858),       6.99  ), 
    (cal_to_jd(1859),       7.19  ), 
    (cal_to_jd(1860),       7.35  ), 
    (cal_to_jd(1861),       7.41  ), 
    (cal_to_jd(1862),       7.36  ), 
    (cal_to_jd(1863),       6.95  ), 
    (cal_to_jd(1864),       6.45  ), 
    (cal_to_jd(1865),       5.92  ), 
    (cal_to_jd(1866),       5.15  ), 
    (cal_to_jd(1867),       4.11  ), 
    (cal_to_jd(1868),       2.94  ), 
    (cal_to_jd(1869),       1.97  ), 
    (cal_to_jd(1870),       1.04  ), 
    (cal_to_jd(1871),       0.11  ), 
    (cal_to_jd(1872),      -0.82  ), 
    (cal_to_jd(1873),      -1.70  ), 
    (cal_to_jd(1874),      -2.48  ), 
    (cal_to_jd(1875),      -3.19  ), 
    (cal_to_jd(1876),      -3.84  ), 
    (cal_to_jd(1877),      -4.43  ), 
    (cal_to_jd(1878),      -4.79  ), 
    (cal_to_jd(1879),      -5.09  ), 
    (cal_to_jd(1880),      -5.36  ), 
    (cal_to_jd(1881),      -5.37  ), 
    (cal_to_jd(1882),      -5.34  ), 
    (cal_to_jd(1883),      -5.40  ), 
    (cal_to_jd(1884),      -5.58  ), 
    (cal_to_jd(1885),      -5.74  ), 
    (cal_to_jd(1886),      -5.69  ), 
    (cal_to_jd(1887),      -5.67  ), 
    (cal_to_jd(1888),      -5.73  ), 
    (cal_to_jd(1889),      -5.78  ), 
    (cal_to_jd(1890),      -5.86  ), 
    (cal_to_jd(1891),      -6.01  ), 
    (cal_to_jd(1892),      -6.28  ), 
    (cal_to_jd(1893),      -6.53  ), 
    (cal_to_jd(1894),      -6.50  ), 
    (cal_to_jd(1895),      -6.41  ), 
    (cal_to_jd(1896),      -6.11  ), 
    (cal_to_jd(1897),      -5.63  ), 
    (cal_to_jd(1898),      -4.68  ), 
    (cal_to_jd(1899),      -3.72  ), 
    (cal_to_jd(1900),      -2.70  ), 
    (cal_to_jd(1901),      -1.48  ), 
    (cal_to_jd(1902),      -0.08  ), 
    (cal_to_jd(1903),       1.26  ), 
    (cal_to_jd(1904),       2.59  ), 
    (cal_to_jd(1905),       3.92  ), 
    (cal_to_jd(1906),       5.20  ), 
    (cal_to_jd(1907),       6.29  ), 
    (cal_to_jd(1908),       7.68  ), 
    (cal_to_jd(1909),       9.13  ), 
    (cal_to_jd(1910),      10.38  ), 
    (cal_to_jd(1911),      11.64  ), 
    (cal_to_jd(1912),      13.23  ), 
    (cal_to_jd(1913),      14.69  ), 
    (cal_to_jd(1914),      16.00  ), 
    (cal_to_jd(1915),      17.19  ), 
    (cal_to_jd(1916),      18.19  ), 
    (cal_to_jd(1917),      19.13  ), 
    (cal_to_jd(1918),      20.14  ), 
    (cal_to_jd(1919),      20.86  ), 
    (cal_to_jd(1920),      21.41  ), 
    (cal_to_jd(1921),      22.06  ), 
    (cal_to_jd(1922),      22.51  ), 
    (cal_to_jd(1923),      23.01  ), 
    (cal_to_jd(1924),      23.46  ), 
    (cal_to_jd(1925),      23.63  ), 
    (cal_to_jd(1926),      23.95  ), 
    (cal_to_jd(1927),      24.39  ), 
    (cal_to_jd(1928),      24.34  ), 
    (cal_to_jd(1929),      24.10  ), 
    (cal_to_jd(1930),      24.02  ), 
    (cal_to_jd(1931),      23.98  ), 
    (cal_to_jd(1932),      23.89  ), 
    (cal_to_jd(1933),      23.93  ), 
    (cal_to_jd(1934),      23.88  ), 
    (cal_to_jd(1935),      23.91  ), 
    (cal_to_jd(1936),      23.76  ), 
    (cal_to_jd(1937),      23.91  ), 
    (cal_to_jd(1938),      23.96  ), 
    (cal_to_jd(1939),      24.04  ), 
    (cal_to_jd(1940),      24.35  ), 
    (cal_to_jd(1941),      24.82  ), 
    (cal_to_jd(1942),      25.30  ), 
    (cal_to_jd(1943),      25.77  ), 
    (cal_to_jd(1944),      26.27  ), 
    (cal_to_jd(1945),      26.76  ), 
    (cal_to_jd(1946),      27.27  ), 
    (cal_to_jd(1947),      27.77  ), 
    (cal_to_jd(1948),      28.25  ), 
    (cal_to_jd(1949),      28.70  ), 
    (cal_to_jd(1950),      29.15  ), 
    (cal_to_jd(1951),      29.57  ), 
    (cal_to_jd(1952),      29.97  ), 
    (cal_to_jd(1953),      30.36  ), 
    (cal_to_jd(1954),      30.72  ), 
    (cal_to_jd(1955),      31.07  ), 
    (cal_to_jd(1956),      31.349 ), 
    (cal_to_jd(1957),      31.677 ), 
    (cal_to_jd(1958),      32.166 ), 
    (cal_to_jd(1959),      32.671 ), 
    (cal_to_jd(1960),      33.150 ), 
    (cal_to_jd(1961),      33.584 ), 
    (cal_to_jd(1962),      33.992 ), 
    (cal_to_jd(1963),      34.466 ), 
    (cal_to_jd(1964),      35.030 ), 
    (cal_to_jd(1965),      35.738 ), 
    (cal_to_jd(1966),      36.546 ), 
    (cal_to_jd(1967),      37.429 ), 
    (cal_to_jd(1968),      38.291 ), 
    (cal_to_jd(1969),      39.204 ), 
    (cal_to_jd(1970),      40.182 ), 
    (cal_to_jd(1971),      41.170 ), 
    (cal_to_jd(1972),      42.227 ), 
    (cal_to_jd(1973),      43.373 ), 
    (cal_to_jd(1974),      44.4841), 
#   (cal_to_jd(1974),      44.486 ), 
    (cal_to_jd(1975),      45.4761), 
#   (cal_to_jd(1975),      45.477 ), 
    (cal_to_jd(1976),      46.4567), 
#   (cal_to_jd(1976),      46.458 ), 
    (cal_to_jd(1977),      47.5214), 
#   (cal_to_jd(1977),      47.521 ), 
    (cal_to_jd(1978),      48.5344), 
#   (cal_to_jd(1978),      48.535 ), 
    (cal_to_jd(1979),      49.5861), 
#   (cal_to_jd(1979),      49.589 ), 
    (cal_to_jd(1980),      50.5387), 
#   (cal_to_jd(1980),      50.540 ), 
    (cal_to_jd(1981),      51.3808), 
#   (cal_to_jd(1981),      51.382 ), 
    (cal_to_jd(1982),      52.1668), 
#   (cal_to_jd(1982),      52.168 ), 
    (cal_to_jd(1983),      52.9565), 
#   (cal_to_jd(1983),      52.957 ), 
    (cal_to_jd(1984),      53.7882), 
#   (cal_to_jd(1984),      53.789 ), 
    (cal_to_jd(1985),      54.3427), 
    (cal_to_jd(1986),      54.8712), 
    (cal_to_jd(1987),      55.3222), 
    (cal_to_jd(1988),      55.8197), 
    (cal_to_jd(1989),      56.3000), 
    (cal_to_jd(1990),      56.8553), 
    (cal_to_jd(1991),      57.5653), 
    (cal_to_jd(1992),      58.3092), 
    (cal_to_jd(1993),      59.1218), 
    (cal_to_jd(1994),      59.9845), 
    (cal_to_jd(1995),      60.7853), 
    (cal_to_jd(1996),      61.6287), 
    (cal_to_jd(1997),      62.2950), 
    (cal_to_jd(1998),      62.9659), 
    (cal_to_jd(1999),      63.4673), 
    (cal_to_jd(2000),      63.8285), 
    (cal_to_jd(2001),      64.0908), 
    (cal_to_jd(2002),      64.2998), 
    (cal_to_jd(2003),      64.4734), 
    (cal_to_jd(2004),      64.5736), 
    (cal_to_jd(2005),      64.6876), 
    (cal_to_jd(2006),      64.8452), 
    (cal_to_jd(2007),      65.1464), 
    (cal_to_jd(2008),      65.4574), 
    (cal_to_jd(2009),      65.7768), 
    (cal_to_jd(2010),      66.0699), 

# Predicted values! Replace after measured are available...
    (cal_to_jd(2011),      67.1   ), 
    (cal_to_jd(2012),      68.    ), 
    (cal_to_jd(2013),      68.    ), 
    (cal_to_jd(2014),      69.    ), 
    (cal_to_jd(2015),      69.    ), 
    (cal_to_jd(2016),      70.    ), 
    (cal_to_jd(2017),      70.    ), 

  ]
_tbl_start = 1620
_tbl_end = 2017


def deltaT_seconds(jd):
    """Return deltaT as seconds of time.

    For a historical range from 1620 to a recent year, we interpolate from a
    table of observed values. Outside that range we use formulae.

    Parameters:
        jd : Julian Day number
    Returns:
        deltaT in seconds

    """
    yr, mo, day = jd_to_cal(jd)
    #
    # 1620 - 20xx
    #
    if _tbl_start < yr < _tbl_end:
        idx = bisect(_tbl, (jd, 0))
        jd1, secs1 = _tbl[idx]
        jd0, secs0 = _tbl[idx-1]
        # simple linear interpolation between two values
        return ((jd - jd0) * (secs1 - secs0) / (jd1 - jd0)) + secs0

    t = (yr - 2000) / 100.0

    #
    # before 948 [Meeus-1998: equation 10.1]
    #
    if yr < 948:
        return polynomial([2177, 497, 44.1], t)

    #
    # 948 - 1620 and after 2000 [Meeus-1998: equation 10.2)
    #
    result = polynomial([102, 102, 25.3], t)

    #
    # correction for 2000-2100 [Meeus-1998: pg 78]
    #
    if _tbl_end < yr < 2100:
        result += 0.37 * (yr - 2100)
    return result


def dt_to_ut(jd):
    """Convert Julian Day from dynamical to universal time.

    Parameters:
        jd : Julian Day number (dynamical time)
    Returns:
        Julian Day number (universal time)

    """
    return jd - deltaT_seconds(jd) / seconds_per_day


