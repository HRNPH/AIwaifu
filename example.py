from vtube_studio import Char_control
if __name__ == "__main__":
    vtube = Char_control()
    while True:
        express = input("Enter expression: ")
        print("Sending expression...")
        vtube.express(express)