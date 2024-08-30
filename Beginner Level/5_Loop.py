# --------------------------------------------------------------------
# 5 For Loop
print('-----------5. For Loop----------')
# --------------------------------------------------------------------

# 5.1 simulate rolling a six-sided die multiple.
print('------ 5.1 simulate rolling a six-sided die multiple ------')

import random

count_6 = 0
count_1 = 0
count_two_6s_in_row = 0
num_of_rolls = 20
previous_roll = None

for _ in range(num_of_rolls):
    roll = random.randint(1, 6)

    if roll == 6:
        count_6 += 1

    elif roll == 1:
        count_1 += 1

    elif roll == 6 and previous_roll == 6:
        count_two_6s_in_row += 1

    previous_roll = roll


print(f'Number of times 6 was rolled : {count_6}')
print(f'Number of times 1 was rolled : {count_1}')
print(f'Number of times two consecutive 6s were rolled : {count_two_6s_in_row}')

# -----------------------------------------------------------------
# 5.2 Workout count for jumping jacks
print('-------- Workout count for jumping jacks --------')

total_jumping_jacks = 100
jumping_jacks_per_set = 10

for completed_jumping_jacks in range(jumping_jacks_per_set, total_jumping_jacks + 1, jumping_jacks_per_set):
    print(f'You have completed {completed_jumping_jacks} jumping jacks.')

    if completed_jumping_jacks >= total_jumping_jacks:
        print(f'Congratulation! You Completed the workout.')
        break

    tired = input('Are you tired? (yes/y or no/n) : ').strip().lower()

    if tired in ['yes', 'y']:

        skip = input('Do you want to skip the remaining set? (yes/y or no/n)').strip().lower()

        if skip in ['yes', 'y']:
            print(f'You completed a total of {completed_jumping_jacks} jumping jacks.')
            break

    else:
        remaining_jumping_jacks = total_jumping_jacks - completed_jumping_jacks
        print(f'{remaining_jumping_jacks} jumping jacks remaining.')
