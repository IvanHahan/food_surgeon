
from pydantic import BaseModel, Field


# Define the Dish model to structure the response for a single dish
class Dish(BaseModel):
    """Always use this tool to structure your response to the user."""

    id: str = Field(description="Identifier of the dish from the retrieved data.")
    name: str = Field(
        description="Name of the dish refined and translated to ukrainian."
    )
    type: str = Field(
        description="Type of the dish refined and translated to ukrainian."
    )
    ingredients: str = Field(
        description="Ingredients of the dish from the retrieved data."
    )
    description: str = Field(
        description="Steps to prepare the dish from the retrieved data. You must always rephrase and enrich it yourself"
    )
    comments: str = Field(
        description="You must always add your personal thoughts on recipe here."
    )

# Define the DishList model to structure the response when multiple dishes are retrieved
class DishList(BaseModel):
    """Always use this tool to structure your response to the user if you have several dishes as output. Put empty list, if no relevant dish found"""

    dishes: list[Dish] = Field(description="List of dishes.")
