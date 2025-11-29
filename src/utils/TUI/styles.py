import questionary

def get_custom_style():
    return questionary.Style([
        ('qmark', 'fg:#673ab7 bold'),       # Token.QuestionMark
        ('question', 'bold'),               # Token.Question
        ('answer', 'fg:#f44336 bold'),      # Token.Answer
        ('pointer', 'fg:#673ab7 bold'),     # Token.Pointer
        ('highlighted', 'fg:#673ab7 bold'), # Token.Highlighted
        ('selected', 'fg:#cc5454'),         # Token.Selected
        ('separator', 'fg:#cc5454'),        # Token.Separator
        ('instruction', '')                 # Token.Instruction
    ])
