def optionMenu():
    choice = input("""
        For circle detection, please select one of the following methods:

        A: Simple Shape Detection using Contour approximation
        B: Circle Hough Transform

        ==> """)

    choice = choice.lower()

    if choice != "a" and choice != "b":
        print("\n\tYou must only select either A or B")
        print("\tPlease try again!\n")
        optionMenu()

    return choice == "b"