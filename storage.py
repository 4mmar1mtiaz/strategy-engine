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


ADMIN_UID = "adminman"


def load_winners(user_id: str) -> list:
    try:
        if user_id == ADMIN_UID:
            result = (
                _client()
                .table("winners")
                .select("user_id, data")
                .execute()
            )
            # Tag each winner with who found it
            winners = []
            for row in result.data:
                entry = row["data"]
                entry["_user_id"] = row["user_id"]
                winners.append(entry)
            return winners
        else:
            result = (
                _client()
                .table("winners")
                .select("data")
                .eq("user_id", user_id)
                .execute()
            )
            return [row["data"] for row in result.data]
    except Exception as e:
        st.warning(f"Could not load saved winners: {e}")
        return []


def delete_winners(user_id: str):
    _client().table("winners").delete().eq("user_id", user_id).execute()
