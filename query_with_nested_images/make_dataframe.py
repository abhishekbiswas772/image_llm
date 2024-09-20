import pandas as pd
 
data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "partname": [
        "Distributor", "Distributor O-Ring", "Cylinder Head Cover", "Cylinder Head Cover Gasket",
        "Rubber Grommet", "Cylinder Head", "Head Gasket", "Camshaft Pulley", "Intake Manifold",
        "Intake Manifold Gasket", "Oil Filter", "Water Pump", "Water Pump Gasket", "Timing Belt Drive Pulley",
        "Oil Pan", "Oil Pan Gasket", "Drain Bolt Crush Washer", "Oil Pan Drain Bolt", "Exhaust Manifold",
        "Exhaust Manifold Gasket", "Engine Block"
    ],
    "description": [
        "Distributes electrical current to the spark plugs in the correct firing order.",
        "Seals the connection between the distributor and engine block.",
        "Protects the top of the engine and valve train.",
        "Seals the cover to the cylinder head.",
        "Provides a seal and reduces vibration for various components.",
        "Contains the combustion chambers and valve train.",
        "Seals the junction between the cylinder head and engine block.",
        "Drives the camshaft via the timing belt.",
        "Distributes air or air/fuel mixture to the cylinders.",
        "Seals the intake manifold to the cylinder head.",
        "Removes contaminants from engine oil.",
        "Circulates coolant through the engine.",
        "Seals the water pump to the engine block.",
        "Drives the timing belt, which synchronizes engine components.",
        "Holds the engine oil reservoir.",
        "Seals the oil pan to the engine block.",
        "Provides a seal for the oil drain bolt.",
        "Allows for oil to be drained from the engine.",
        "Collects exhaust gases from the cylinders.",
        "Seals the exhaust manifold to the cylinder head.",
        "The main body of the engine that houses the cylinders and crankshaft."
    ]
}
 
df = pd.DataFrame(data)
df.to_csv('car_engine_parts.csv', index=False)
print("DataFrame has been saved to 'car_engine_parts.csv'.")