import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def _client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def save_winner(user_id: str, winner: dict):
    _client().table("winners").insert({
        "user_id": user_id,
        "data": winner,
    }).execute()


def load_winners(user_id: str) -> list:
    result = (
        _client()
        .table("winners")
        .select("data")
        .eq("user_id", user_id)
        .order("saved_at")
        .execute()
    )
    return [row["data"] for row in result.data]


def delete_winners(user_id: str):
    _client().table("winners").delete().eq("user_id", user_id).execute()
