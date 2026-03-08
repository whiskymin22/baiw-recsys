from typing import List, Set

import pandas as pd
from pymongo import MongoClient

from .base import BaseDatasetLoader
from .constants import MONGO_DB, MONGO_URI


class MongoDatasetLoader(BaseDatasetLoader):
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def load(self, *, collection_name: str = "Purchases"):
        """Load interactions from MongoDB (Purchases, Wishlists, ProductViews)."""
        records = []

        for doc in self.db["Purchases"].find({"user_id": {"$ne": None}}):
            for item in doc.get("items", []):
                records.append({
                    "user_id": str(doc.get("user_id")),
                    "item_id": str(item.get("product_id")),
                    "interaction": 3,
                    "timestamp": doc.get("createdAt"),
                })

        for doc in self.db["Wishlists"].find({"user_id": {"$ne": None}}):
            records.append({
                "user_id": str(doc.get("user_id")),
                "item_id": str(doc.get("product_id")),
                "interaction": 2,
                "timestamp": doc.get("createdAt"),
            })

        for doc in self.db["ProductViews"].find({"user_id": {"$ne": None}}):
            records.append({
                "user_id": str(doc.get("user_id")),
                "item_id": str(doc.get("product_id")),
                "interaction": 1,
                "timestamp": doc.get("viewedAt"),
            })

        if not records:
            df = pd.DataFrame(columns=["user_id", "item_id", "interaction", "timestamp"])
        else:
            df = pd.DataFrame(records)

        class SimpleDataset:
            def __init__(self, pandas_df):
                self._df = pandas_df
            def get_pandas_dataframe(self):
                return self._df

        return SimpleDataset(df)

    def load_courses_popularity(self, n: int = 100):
        """Load popular products based on Purchases, Wishlists, and ProductViews.
        
        Scoring:
        - Each purchase = 3 points (highest intent)
        - Each wishlist = 2 points
        - Each view = 1 point
        """
        PURCHASE_WEIGHT = 3
        WISHLIST_WEIGHT = 2
        VIEW_WEIGHT = 1
        
        product_scores: dict[str, float] = {}
        
        # Count purchases per product
        purchase_pipeline = [
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "count": {"$sum": 1}}},
        ]
        try:
            purchase_cursor = self.db["Purchases"].aggregate(purchase_pipeline)
            for doc in purchase_cursor:
                product_id = str(doc["_id"])
                product_scores[product_id] = product_scores.get(product_id, 0) + doc["count"] * PURCHASE_WEIGHT
        except Exception:
            pass
        
        # Count wishlist per product
        wishlist_pipeline = [
            {"$match": {"product_id": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": "$product_id", "count": {"$sum": 1}}},
        ]
        try:
            wishlist_cursor = self.db["Wishlists"].aggregate(wishlist_pipeline)
            for doc in wishlist_cursor:
                product_id = str(doc["_id"])
                product_scores[product_id] = product_scores.get(product_id, 0) + doc["count"] * WISHLIST_WEIGHT
        except Exception:
            pass
        
        # Count views per product
        view_pipeline = [
            {"$match": {"product_id": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": "$product_id", "count": {"$sum": 1}}},
        ]
        try:
            view_cursor = self.db["ProductViews"].aggregate(view_pipeline)
            for doc in view_cursor:
                product_id = str(doc["_id"])
                product_scores[product_id] = product_scores.get(product_id, 0) + doc["count"] * VIEW_WEIGHT
        except Exception:
            pass
        
        sorted_products = sorted(
            product_scores.items(), key=lambda x: x[1], reverse=True
        )[:n]
        
        if not sorted_products:
            df = pd.DataFrame(columns=["item_id", "interaction"])
        else:
            records = [
                {"item_id": pid, "interaction": score}
                for pid, score in sorted_products
            ]
            df = pd.DataFrame(records)
            
        class SimpleDataset:
            def get_pandas_dataframe(self):
                return df
        return SimpleDataset()

    def load_products_by_tags(self, max_transaction_size: int = 20):
        """Load products grouped by tags for FP-Growth transactions."""
        cursor = (
            self.db["Products"]
            .find(
                {"tags": {"$exists": True, "$ne": []}},
                {"_id": 1, "tags": 1},
            )
            .limit(500)
        )

        tag_to_products = {}
        all_products = []
        for doc in cursor:
            product_id = str(doc["_id"])
            tags = doc.get("tags", [])
            all_products.append((product_id, tags))
            for tag in tags:
                if tag not in tag_to_products:
                    tag_to_products[tag] = set()
                tag_to_products[tag].add(product_id)

        transactions = []
        for product_id, tags in all_products:
            related = set()
            for tag in tags:
                related.update(tag_to_products.get(tag, set()))
            related.discard(product_id)
            if len(related) >= 2:
                transaction = [product_id] + list(related)[: max_transaction_size - 1]
                transactions.append(transaction)

        return transactions

    def load_user_baskets(self) -> List[List[str]]:
        """Load user baskets from Cart and Purchases for FP-Growth."""
        transactions = []

        # From Carts - each user's cart items form a basket
        try:
            cart_pipeline = [
                {"$match": {"items": {"$exists": True, "$ne": []}}},
                {"$project": {"products": "$items.product_id"}},
            ]
            cart_cursor = self.db["Carts"].aggregate(cart_pipeline)
            for doc in cart_cursor:
                products = [str(p) for p in doc.get("products", []) if p]
                if len(products) >= 2:
                    transactions.append(products)
        except Exception:
            pass

        # From Purchases
        try:
            purchase_pipeline = [
                {"$match": {"items": {"$exists": True, "$ne": []}}},
                {"$project": {"products": "$items.product_id"}},
            ]
            purchase_cursor = self.db["Purchases"].aggregate(purchase_pipeline)
            for doc in purchase_cursor:
                products = [str(p) for p in doc.get("products", []) if p]
                if len(products) >= 2:
                    transactions.append(products)
        except Exception:
            pass

        # Fallback: ProductViews - group by user_id
        if len(transactions) < 10:
            try:
                view_pipeline = [
                    {"$match": {"user_id": {"$exists": True, "$ne": None}}},
                    {"$group": {"_id": "$user_id", "products": {"$addToSet": "$product_id"}}},
                    {"$match": {"products.1": {"$exists": True}}},
                ]
                view_cursor = self.db["ProductViews"].aggregate(view_pipeline)
                for doc in view_cursor:
                    products = [str(p) for p in doc.get("products", []) if p]
                    if len(products) >= 2:
                        transactions.append(products)
            except Exception:
                pass

        return transactions

    def get_user_products(self, user_id: str) -> List[str]:
        """Get all products a user has purchased, in cart, or viewed."""
        from bson import ObjectId

        product_timestamps = {}

        query_variants = [{"user_id": user_id}]
        try:
            query_variants.append({"user_id": ObjectId(user_id)})
        except Exception:
            pass

        def _update_ts(p_id, ts):
            if not p_id:
                return
            p_id = str(p_id)
            ts_val = 0
            if ts:
                if hasattr(ts, "timestamp"):
                    ts_val = ts.timestamp()
                else:
                    try:
                        ts_val = float(ts)
                    except (ValueError, TypeError):
                        ts_val = 0
            if p_id not in product_timestamps or ts_val > product_timestamps[p_id]:
                product_timestamps[p_id] = ts_val

        # Purchases
        for q in query_variants:
            for doc in self.db["Purchases"].find(q):
                for item in doc.get("items", []):
                    _update_ts(item.get("product_id"), doc.get("createdAt"))

        # Cart
        for q in query_variants:
            for doc in self.db["Carts"].find(q):
                for item in doc.get("items", []):
                    _update_ts(item.get("product_id"), doc.get("updatedAt"))

        # ProductViews
        for q in query_variants:
            for doc in self.db["ProductViews"].find(q):
                _update_ts(doc.get("product_id"), doc.get("viewedAt"))

        sorted_products = sorted(
            product_timestamps.items(), key=lambda x: x[1], reverse=True
        )
        return [c[0] for c in sorted_products]
