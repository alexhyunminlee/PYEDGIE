from ochre.Simulator import Simulator


class Building(Simulator):  # type: ignore[no-any-unimported]
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


if __name__ == "__main__":
    building = Building("test_building")
