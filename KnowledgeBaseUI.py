from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, Button, Header, Label
from textual.containers import Container, Grid
from datetime import datetime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from textual_fspicker import FileOpen, Filters
from pathlib import Path
import pandas as pd

Base = declarative_base()


class Knowledge(Base):
    __tablename__ = 'knowledge'

    id = Column(Integer, primary_key=True)
    tmstmp = Column(DateTime)
    document = Column(String)

    def __repr__(self):
        return (f"<Knowledge("
                f"tmstmp='{str(self.tmstmp)}', "
                f"document='{str(self.document)}')>")


# Inherit from the Textual App class
class KnowledgeBase(App):
    CSS_PATH = "KnowledgeBase/style.tcss"

    def __init__(self, session):
        super().__init__()
        self.session = session  # Store the session for DB access
        self.data_table = None
        self.logger = None

    def compose(self) -> ComposeResult:
        # Compose the widgets in the UI
        # Vertical layout for placing elements above the table

        yield Header(True)

        with Container():  # Create a container for the buttons
            yield Input(placeholder="Enter document", id="txt-doc")
            with Grid(id="buttons"):
                yield Button("From CSV", id="btn-from-csv")
                yield Button("Add Document", id="btn-add-doc")
                yield Button("Remove Document", id="btn-remove")
                yield Button("Quit", id="btn-exit")
            self.logger = Label("Operation Result: ", id="lbl-logger")
            yield self.logger

        # Now add the DataTable below
        self.data_table = DataTable(id="data_table")  # Store a reference to the DataTable
        yield self.data_table

    async def on_mount(self):
        # Called when the app is first mounted
        data_table = self.query_one(DataTable)
        data_table.add_column("ID", width=5)
        data_table.add_column("Document", width=120)
        data_table.cursor_type = "row"

        # Load initial data into the table
        self.load_data()

    def update_logger(self, msg: str):
        self.logger.update(f"Operation Result: {msg}")

    def process_csv(self, to_show: Path | None) -> None:
        if to_show is None:
            self.update_logger("Cancelled")
            return None
        df = pd.read_csv(str(to_show), sep=";")
        for index, row in df.iterrows():
            new_entry = Knowledge(
                tmstmp=datetime.now(),
                document=row['assistant']
            )
            self.session.add(new_entry)
        self.session.commit()
        self.load_data()

    def load_data(self):
        """Load data from the database and display in the DataTable."""
        # Clear any existing rows
        self.data_table.clear()

        # Query all rows from the knowledge table
        rows = self.session.query(Knowledge).all()

        # Add rows to the DataTable
        for row in rows:
            self.data_table.add_row(str(row.id), row.document)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when the add button is pressed."""
        if event.button.id == "btn-add-doc":
            # Get input data from the text field
            document_input = self.query_one("#txt-doc", Input).value

            # Validate if the input is not empty
            if document_input.strip():
                # Add new document to the database
                new_entry = Knowledge(
                    tmstmp=datetime.now(),
                    document=document_input
                )
                self.session.add(new_entry)
                self.session.commit()

                # Refresh the table with updated data
                self.load_data()

                # Clear the input field
                self.query_one("#txt-doc", Input).value = ""
                self.update_logger("Addition Completed!")

        elif event.button.id == "btn-from-csv":
            await self.push_screen(
                FileOpen(
                    "KnowledgeBase",
                    filters=Filters(
                        ("CSV", lambda p: p.suffix.lower() == ".csv"),
                    ),
                ),
                callback=self.process_csv,
            )

        elif event.button.id == "btn-remove":

            # Get the selected row index
            self.selected_row = self.data_table.cursor_row  # The current selected row index

            if self.selected_row is not None:
                # Get the ID of the selected row
                self.selected_row_data = self.data_table.get_row_at(self.selected_row)

                # Query the specific entry to remove
                entry_to_remove = self.session.query(Knowledge).filter_by(id=int(self.selected_row_data[0])).first()

                if entry_to_remove:
                    self.session.delete(entry_to_remove)
                    self.session.commit()
                    self.load_data()  # Refresh the table
                self.update_logger("Deletion Completed!")

        elif event.button.id == "btn-exit":
            # Exit the application
            await self.action_quit()


if __name__ == "__main__":
    engine = create_engine("sqlite:///API_Database.db", echo=False)
    session = sessionmaker(bind=engine)
    app = KnowledgeBase(session())
    app.run()