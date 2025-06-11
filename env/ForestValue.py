from typing import Dict

price_table = {
    "costs": {
        "admin_cost": 50,
        "thinning_cost": 150,
        "harvesting_cost": 75,
        "seedling_cost": 150,
        "planting_cost": 70,
        "other_cost": 150
    },
    "revenues": {
        "interest_rate": 0.04,
        "hunting_income": 20,
        "pulp_price": 100,
        "cns_price": 100,
        "saw_price": 100,
    },
}

product_specs_table = {
    "pulp_diameter_max": 5,
    "cns_diameter_max": 7,
    "saw_diameter_max": 12,
}


class PriceTable:
    """
    Holds fixed unit costs and revenues associated with stand-level forest operations and products.
    """
    def __init__(self, price_data: dict):
        self.costs = price_data.get("costs", {})
        self.revenues = price_data.get("revenues", {})

    def cost(self, key: str) -> float:
        return self.costs.get(key, 0.0)

    def price(self, product: str) -> float:
        return self.revenues.get(f"{product}_price", 0.0)

    def interest_rate(self) -> float:
        return self.revenues.get("interest_rate", 0.04)

    def hunting_income(self) -> float:
        return self.revenues.get("hunting_income", 0.0)


class CashFlowLedger:
    """
    Records all forest management-related cash flows for a given rotation.
    """
    def __init__(self):
        self.cash_flows: Dict[int, float] = {}

    def add_cost(self, year: int, amount: float):
        self.cash_flows[year] = self.cash_flows.get(year, 0.0) - abs(amount)

    def add_revenue(self, year: int, amount: float):
        self.cash_flows[year] = self.cash_flows.get(year, 0.0) + amount

    def add_annual_payment(self, start_year: int, end_year: int, amount: float):
        for year in range(start_year, end_year + 1):
            self.add_revenue(year, amount) if amount > 0 else self.add_cost(year, -amount)


class LEVModel:
    """
    Computes the Land Expectation Value (LEV) for a rotation of timber production.
    """
    def __init__(self, price_table: PriceTable, ledger: CashFlowLedger):
        self.prices = price_table
        self.ledger = ledger

    def compute_nfv(self, rotation_length: int) -> float:
        """
        Computes the Net Future Value (NFV) of a single rotation by compounding each cash flow to the end of rotation.
        """
        i = self.prices.interest_rate()
        return sum(
            amount * ((1 + i) ** (rotation_length - year))
            for year, amount in self.ledger.cash_flows.items()
        )

    def compute_lev(self, rotation_length: int) -> float:
        """
        Calculates the LEV using Faustmann’s formula for infinite identical rotations:
        LEV = NFV / ((1 + i)^t - 1)
        """
        nfv = self.compute_nfv(rotation_length)
        i = self.prices.interest_rate()
        denominator = (1 + i) ** rotation_length - 1
        return nfv / denominator if denominator != 0 else float("-inf")


def optimal_rotation() -> int:
    """
    Placeholder function for determining the biologically or economically optimal rotation length.
    """
    return 30  # TODO: Implement this


"""
Example usage:

prices = PriceTable(price_table)
ledger = CashFlowLedger()

rotation_length = optimal_rotation()

# Record one-time establishment costs
ledger.add_cost(0, prices.cost("planting_cost") + prices.cost("seedling_cost"))

# Add simulated thinning and final harvest revenues
ledger.add_revenue(18, 12 * prices.price("pulp"))  # thinning volume
ledger.add_revenue(25, 5 * prices.price("cns"))
ledger.add_revenue(30, 20 * prices.price("saw"))

# Add recurring annual income (e.g., hunting)
ledger.add_annual_payment(1, rotation_length, prices.hunting_income())

model = LEVModel(prices, ledger)
print("NFV:", model.compute_nfv(rotation_length))
print("LEV:", model.compute_lev(rotation_length))

###

i.e. in RL environment
if action == "thin":
    ledger.add_cost(t, prices.cost("thinning_cost"))
    ledger.add_revenue(t, vol_pulp * prices.price("pulp"))
"""
