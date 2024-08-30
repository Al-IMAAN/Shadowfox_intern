# -------------------------------------------------
# 2. Numbers
print('---------- 2. Numbers ----------')
# --------------------------------------------------

# 2.1 Write a function that takes two arguments, 145 and 'o',
# and uses the `format` function to return a formatted string.
# Print the result.
# Try to identify the representation used.


def format_example(number, format_spec):
    return f'The Formatted string is {format(number, format_spec)}'


print(format_example(145, 'o'))

# The Formatted stings converts the decimal number into octal number

# ----------------------------------------------------
# 2.2 In a village, there is a circular pond with a radius of 84 meters.
# Calculate the area of the pond

rad_of_pond = 84
water_per_square_meter = 1.4
area_of_pond = 3.14 * (rad_of_pond ** 2)
total_water = area_of_pond * water_per_square_meter

print(f'The area of Pond is {round(area_of_pond)} sq')
print(f'The Total water in the pond is {round(total_water)} liter')

# --------------------------------------------------------
# 2.3 If you cross a 490-meter-long street in 7 minutes, calculate your speed in meter per second.
# Print the answer without any decimal point in it.

distance = 490
time = 420
speed = distance / time

print(f'The time taken speed to cover 490 meter long street is {round(speed)} mps')
