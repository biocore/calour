{{ fullname }}
{{ underline }}
.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
   :show-inheritance:

{# Taken from scipy's sphinx documentation setup (https://github.com/scipy/scipy/blob/master/doc/source/_templates/autosummary/class.rst). #}

{% block methods %}
{% if all_methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:
      {% for item in all_methods %}
         {# We want to build dunder methods if they exist, but not every kind of dunder. These are the dunders provided by default on `object` #}
         {%- if not item.startswith('_') or (item not in ['__class__',
                                                          '__weakref__',
                                                          '__delattr__',
                                                          '__getattribute__',
                                                          '__init__',
                                                          '__dir__',
                                                          '__new__',
                                                          '__format__',
                                                          '__reduce__',
                                                          '__reduce_ex__',
                                                          '__setattr__',
                                                          '__sizeof__',
                                                          '__subclasshook__'] and item.startswith('__')) %}
            ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
{% endif %}
{% endblock %}


{% block attributes %}
{% if all_attributes %}
   .. rubric:: Attributes
   .. autosummary::
      :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
{% endif %}
{% endblock %}
