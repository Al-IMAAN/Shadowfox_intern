# ------------------------------------------------------------------------
# 6 Dictionary
print('-----------6. Dictionary ----------')
# --------------------------------------------------------------------

# 6.1 Create a friends name using list and and length of name in dictionary.
print('--------- 6.1 Create a Dictionary using friend_name and add their length ----------')
friends_name = ['Alpha', 'Bob', 'Charlie', 'Dave', 'Ethias']

name_length = {(name, len(name)) for name in friends_name}
print(name_length)

# --------------------------------------------------------------------------------
# 6.2 Calculate expense of you and your wife for trip.
print('---- 6.2 Calculate expense of you and your wife for trip. ----')

my_expense = {
    'Hotel': 1200,
    'Food': 800,
    'Transportation': 500,
    'Attraction': 300,
    'Miscellaneous': 200
}

partner_expense = {
    'Hotel': 1000,
    'Food': 900,
    'Transportation': 600,
    'Attraction': 400,
    'Miscellaneous': 150
}

total_of_my_expense = sum(my_expense.values())
total_of_partner_expense = sum(partner_expense.values())

if total_of_my_expense > total_of_partner_expense:
    more_spent = 'I spent more money overall.'
elif total_of_my_expense < total_of_partner_expense:
    more_spent = 'My partner spent more money overall.'
else:
    more_spent = 'We both spent the same amount of money.'

significant_difference = None
max_difference = 0

for category in my_expense:
    difference = abs(my_expense[category] - partner_expense[category])
    if difference > max_difference:
        max_difference = difference
        significant_difference = category

print(f'Total My Expense : ${total_of_my_expense}')
print(f"Total Partner's Expense : ${total_of_partner_expense}")
print(more_spent)
print(f'Significant difference in the "{significant_difference}" category with a difference of ${max_difference}')
