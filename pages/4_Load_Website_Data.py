# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from webgpt.tools import  add_knowledge_base_to_store
from streamlit.hello.utils import show_code

st.set_page_config(page_title="Load Website Data", page_icon="📊")
st.markdown("# Website Data")
st.markdown('<p style="color: #0000FF; font-weight: bold;">Your URL\'s:</p>', unsafe_allow_html=True)

url = ['https://testsite.loadcanadauscanada.com/property-owners/','https://testsite.loadcanadauscanada.com/about/','https://testsite.loadcanadauscanada.com/popular-areas/','https://testsite.loadcanadauscanada.com/contact/']

# st.text(url)
for u in url:
    st.markdown(f"[{u}]({u})", unsafe_allow_html=True)
    # st.text(u)
if st.button("Fetch Data"):
    with st.spinner("Processing"):
        add_knowledge_base_to_store()
#st.sidebar.header("Website Data")
with st.sidebar:
    st.subheader("Website Data")



