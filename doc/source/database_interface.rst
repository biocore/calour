Creating database interfaces for Calour
=======================================

Calour is designed with an extendible database interface support. In this document we will describe the steps required to integrate your database into Calour.

Terminology
-----------
Calour handles *Experiments*, which are comprised of a *Sample* X *Feature* table. For example, in a typical deblurred 16S amplicon microbiome experiment, each feature is a unique 16S sequence observed in the samples.

A *Database* contains *annotations* about features, which are keyed by the feature ID, and contain information about the features. Examples of databases (integrated into Calour) include:

* `dbBact <http://dbbact.org>`_ - contains conclusions from experiments where the bacteria were observed (i.e. this bacteria is higher in feces of people with HIV compared to healthy controls). Annotations are manually added to the database.

* `phenoDB <http://msphere.asm.org/content/2/4/e00237-17>`_ - Phenotypic (such as motility, spores, etc.) and environmental (such as temp. and pH preference) information about bacteria (based on Hiding in Plain Sight: Mining Bacterial Species Records for Phenotypic Trait Information, Albert Barberán, Hildamarie Caceres Velazquez, Stuart Jones, Noah Fierer, DOI: 10.1128/mSphere.00237-17)

* `SpongeEMP <http://www.spongeemp.com/main>`_ - Information about marine sponge bacteria, including typical hosts and geographic locations where this bacteria was observed.

The *database interface* is a python module that enables the integration of the database into calour. This can include displaying information about features when they are clicked in the interactive heatmap, performing statistical analysis on the annotations for a set of features and adding database annotations to a selected set of features.

Calour database integration levels
----------------------------------
There are several levels of database integration into calour, depending on the actions the database supports:

L1. Get information about a feature: The database interface gets a unique feature identifier (i.e. the 16S amplicon sequence for the bacteria), and returns a set of strings describing the annotation for the feature. This is used for example in the interactive heatmap - when clicking on a feature, a list of strings describing this feature is shown.

L2. Additional information about a feature: If supported, when double clicking an information string in the interactve heatmap, the database interface handles the GUI to display additional information about this string for the feature. For example, dbBact opens a web browser page with details about the clicked annotation.

L3. Enriched analysis: if supported, given two sets of features, find database annotations that are different between the two feature sets (in a statistically significant manner). This can be accessed both via the interactive heatmap (selecting a group of features and clicking "enrichment") compares this set of features to the rest of the features in the Experiment, or via python commands (such as Experiment.enrichment() )

L4. Add Annotations: if supported, add annotations about selected features to the database. This is used for user-generated annotation databases (such as dbBact). Note that all the GUI for adding the annotation is handled by the database interface. Calour just supplies the Experiment and set of selected features to the database interface.

A database interface declared what integration levels it supports. All database interfaces must at least support level L1.

Writing a Calour database interface
-----------------------------------
In order to create a new database interface, you need to write a python module containing a child class for the calour.database.Database class. In this class, overwrite the relevant functions for the integration level you are supporting. For a minimal implementation (level L1), overwrite the Database.__init__() and the Database.get_seq_annotation_strings() functions.

Additionally, you'll need to add your database to the calour.config file (located in the calour module directory). You need to add a section (named as the desired database name in calour) for your database, containing the module name and the class name for your database. 

Example:

Say we have a fish database we want to add. We create a new module named fish_database, that contains the FishDatabase class (derived from calour.Database). We want to access this database in calour using the name "fishdb".

In the fish_database.py:

.. code-block:: python

   from calour.database import Database

   class FishDatabase(Database):
      def __init__(self, exp=None):
         # we provide methods=['get'] since this database only supports L1 integration
         super().__init__(exp=exp, database_name='fishdb', methods=['get'])

         # and do some database specific setup...
         self.fish_database = READ_THE_DATABASE_TABLE()

      def get_seq_annotation_strings(self, feature):
        '''Get nice string summaries of annotations for a given sequence

        Parameters
        ----------
        feature : str
            the feature unique identifier

        Returns
        -------
        shortdesc : list of (dict,str) (annotationdetails,annotationsummary)
            a list of:
                annotationdetails : dict
                    'seqid' : str, the sequence annotated
                    'annotationtype : str
                    ...(additional keys/values if needed for the database interface use)
                annotationsummary : str
                    a short summary of the annotation
        '''
        shortdesc = []

        # get all annotations from the database and add to the description list
        for annotation in self.GET_DATABASE_ANNOTATIONS(feature):
                shortdesc.append(({'annotationtype': 'other', 'feature': feature}, '%s' % annotation))

        return shortdesc


In the calour.config file we will add the following section::

   [fishdb]
   module_name = fish_database
   class_name = FishDatabase


and thats it.
