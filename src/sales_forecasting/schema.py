"""
Module: schema.py
Description: Restrictions and metadata for sales forecasting dataset
"""

COLUMN_METADATA = {
    "record_ID": {
        "role": "identifier",
        "description": "Artificial row identifier with no business meaning",
        "use_in_model": False,
    },
    "week": {
        "role": "time",
        "description": "Week start date for the aggregated sales period",
        "dtype": "datetime",
    },
    "store_id": {
        "role": "entity_id",
        "description": "Unique identifier for the store",
        "dtype": "category",
    },
    "sku_id": {
        "role": "entity_id",
        "description": "Unique identifier for the product (SKU)",
        "dtype": "category",
    },
    "base_price": {
        "role": "numeric",
        "description": "Regular (non-promotional) unit price",
        "constraints": {
            "min": 0,
            "strict": True
        }
    },
    "total_price": {
        "role": "numeric",
        "description": "Actual average selling unit price during the week",
        "constraints": {
            "min": 0,
            "strict": True
        }
    },
    "is_featured_sku": {
        "role": "promotion_flag",
        "description": "Whether the SKU was featured in marketing promotions",
        "dtype": "category",
        "allowed_values": [0, 1],
    },
    "is_display_sku": {
        "role": "promotion_flag",
        "description": "Whether the SKU had special in-store display",
        "dtype": "category",
        "allowed_values": [0, 1],
    },
    "units_sold": {
        "role": "target",
        "description": "Number of units sold during the week",
        "constraints": {
            "min": 0,
            "strict": False
        }
    },
}
