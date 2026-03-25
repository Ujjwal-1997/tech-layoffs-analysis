{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "50089e27",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/ujjwalkatyal/anaconda3/lib/python3.11/site-packages/transformers/utils/generic.py:260: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  torch.utils._pytree._register_pytree_node(\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from transformers import pipeline\n",
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8a88f5e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>company</th>\n",
       "      <th>layoff_date</th>\n",
       "      <th>jobs_cut</th>\n",
       "      <th>pct_workforce_cut</th>\n",
       "      <th>sector</th>\n",
       "      <th>country</th>\n",
       "      <th>hq_city</th>\n",
       "      <th>ai_cited</th>\n",
       "      <th>reason_stated</th>\n",
       "      <th>company_revenue_2025_bn</th>\n",
       "      <th>...</th>\n",
       "      <th>layoffs_2024</th>\n",
       "      <th>layoffs_2025</th>\n",
       "      <th>verified_source</th>\n",
       "      <th>month</th>\n",
       "      <th>quarter</th>\n",
       "      <th>region</th>\n",
       "      <th>layoff_size_category</th>\n",
       "      <th>stock_reaction</th>\n",
       "      <th>laid_off_vs_headcount_pct</th>\n",
       "      <th>data_as_of</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Amazon</td>\n",
       "      <td>2026-01-15</td>\n",
       "      <td>16000</td>\n",
       "      <td>2.7</td>\n",
       "      <td>E-Commerce/Cloud</td>\n",
       "      <td>USA</td>\n",
       "      <td>Seattle</td>\n",
       "      <td>False</td>\n",
       "      <td>Reduce bureaucracy and management layers</td>\n",
       "      <td>716.9</td>\n",
       "      <td>...</td>\n",
       "      <td>4000</td>\n",
       "      <td>14000</td>\n",
       "      <td>CNBC / NetworkWorld</td>\n",
       "      <td>January 2026</td>\n",
       "      <td>Q1 2026</td>\n",
       "      <td>North America</td>\n",
       "      <td>Mega (5K+)</td>\n",
       "      <td>Positive</td>\n",
       "      <td>1.03</td>\n",
       "      <td>March 18, 2026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Block</td>\n",
       "      <td>2026-02-28</td>\n",
       "      <td>4000</td>\n",
       "      <td>40.0</td>\n",
       "      <td>Fintech</td>\n",
       "      <td>USA</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>True</td>\n",
       "      <td>AI tools replace roles enabling smaller teams</td>\n",
       "      <td>22.4</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1000</td>\n",
       "      <td>CNBC / Crunchbase</td>\n",
       "      <td>February 2026</td>\n",
       "      <td>Q1 2026</td>\n",
       "      <td>North America</td>\n",
       "      <td>Large (2K-5K)</td>\n",
       "      <td>Positive</td>\n",
       "      <td>40.00</td>\n",
       "      <td>March 18, 2026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Meta Reality Labs</td>\n",
       "      <td>2026-01-20</td>\n",
       "      <td>1500</td>\n",
       "      <td>10.0</td>\n",
       "      <td>Social Media/VR</td>\n",
       "      <td>USA</td>\n",
       "      <td>Menlo Park</td>\n",
       "      <td>True</td>\n",
       "      <td>Pivot from metaverse to AI research</td>\n",
       "      <td>164.5</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>500</td>\n",
       "      <td>InformationWeek / NYT</td>\n",
       "      <td>January 2026</td>\n",
       "      <td>Q1 2026</td>\n",
       "      <td>North America</td>\n",
       "      <td>Medium (500-2K)</td>\n",
       "      <td>Positive</td>\n",
       "      <td>1.90</td>\n",
       "      <td>March 18, 2026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Atlassian</td>\n",
       "      <td>2026-03-14</td>\n",
       "      <td>1600</td>\n",
       "      <td>10.0</td>\n",
       "      <td>Enterprise Software</td>\n",
       "      <td>Australia</td>\n",
       "      <td>Sydney</td>\n",
       "      <td>True</td>\n",
       "      <td>Pivot to AI-first company strategy</td>\n",
       "      <td>5.1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>500</td>\n",
       "      <td>TechRepublic / Metaintro</td>\n",
       "      <td>March 2026</td>\n",
       "      <td>Q1 2026</td>\n",
       "      <td>Asia-Pacific</td>\n",
       "      <td>Medium (500-2K)</td>\n",
       "      <td>Positive</td>\n",
       "      <td>10.00</td>\n",
       "      <td>March 18, 2026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Oracle</td>\n",
       "      <td>2026-02-01</td>\n",
       "      <td>30000</td>\n",
       "      <td>15.0</td>\n",
       "      <td>Enterprise Software</td>\n",
       "      <td>USA</td>\n",
       "      <td>Austin</td>\n",
       "      <td>True</td>\n",
       "      <td>AI data centres replace human ops</td>\n",
       "      <td>52.9</td>\n",
       "      <td>...</td>\n",
       "      <td>6000</td>\n",
       "      <td>10000</td>\n",
       "      <td>IBTimes</td>\n",
       "      <td>February 2026</td>\n",
       "      <td>Q1 2026</td>\n",
       "      <td>North America</td>\n",
       "      <td>Mega (5K+)</td>\n",
       "      <td>Positive</td>\n",
       "      <td>15.00</td>\n",
       "      <td>March 18, 2026</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             company layoff_date  jobs_cut  pct_workforce_cut  \\\n",
       "0             Amazon  2026-01-15     16000                2.7   \n",
       "1              Block  2026-02-28      4000               40.0   \n",
       "2  Meta Reality Labs  2026-01-20      1500               10.0   \n",
       "3          Atlassian  2026-03-14      1600               10.0   \n",
       "4             Oracle  2026-02-01     30000               15.0   \n",
       "\n",
       "                sector    country        hq_city  ai_cited  \\\n",
       "0     E-Commerce/Cloud        USA        Seattle     False   \n",
       "1              Fintech        USA  San Francisco      True   \n",
       "2      Social Media/VR        USA     Menlo Park      True   \n",
       "3  Enterprise Software  Australia         Sydney      True   \n",
       "4  Enterprise Software        USA         Austin      True   \n",
       "\n",
       "                                   reason_stated  company_revenue_2025_bn  \\\n",
       "0       Reduce bureaucracy and management layers                    716.9   \n",
       "1  AI tools replace roles enabling smaller teams                     22.4   \n",
       "2            Pivot from metaverse to AI research                    164.5   \n",
       "3             Pivot to AI-first company strategy                      5.1   \n",
       "4              AI data centres replace human ops                     52.9   \n",
       "\n",
       "   ...  layoffs_2024  layoffs_2025           verified_source          month  \\\n",
       "0  ...          4000         14000       CNBC / NetworkWorld   January 2026   \n",
       "1  ...             0          1000         CNBC / Crunchbase  February 2026   \n",
       "2  ...             0           500     InformationWeek / NYT   January 2026   \n",
       "3  ...             0           500  TechRepublic / Metaintro     March 2026   \n",
       "4  ...          6000         10000                   IBTimes  February 2026   \n",
       "\n",
       "   quarter         region  layoff_size_category  stock_reaction  \\\n",
       "0  Q1 2026  North America            Mega (5K+)        Positive   \n",
       "1  Q1 2026  North America         Large (2K-5K)        Positive   \n",
       "2  Q1 2026  North America       Medium (500-2K)        Positive   \n",
       "3  Q1 2026   Asia-Pacific       Medium (500-2K)        Positive   \n",
       "4  Q1 2026  North America            Mega (5K+)        Positive   \n",
       "\n",
       "  laid_off_vs_headcount_pct      data_as_of  \n",
       "0                      1.03  March 18, 2026  \n",
       "1                     40.00  March 18, 2026  \n",
       "2                      1.90  March 18, 2026  \n",
       "3                     10.00  March 18, 2026  \n",
       "4                     15.00  March 18, 2026  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "layoffs_df = pd.read_csv('tech_layoffs_2026_tracker.csv')\n",
    "layoffs_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "622a0c4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No model was supplied, defaulted to facebook/bart-large-mnli and revision c626438 (https://huggingface.co/facebook/bart-large-mnli).\n",
      "Using a pipeline without specifying a model name and revision in production is not recommended.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aafdddd0ed7c4a7e81a9eee31c819052",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading config.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7e75491d958549bba3a7ce128927de2c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading model.safetensors:   0%|          | 0.00/1.63G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5ee079439c974443bcc3fecd72593547",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5107a03195df4ac1a67d5a175ed97424",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading vocab.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0025955ec53a4e05a99c10de61d7da9b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading merges.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2ebc13120e734c368b74de924a7ef9a6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading tokenizer.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "classifier = pipeline(\"zero-shot-classification\", framework='pt')\n",
    "\n",
    "labels = [\"AI\", \"Cost Cutting\", \"Restructuring\", \"Acquisition\", \"Other\"]\n",
    "\n",
    "def classify_reason(text):\n",
    "    result = classifier(text, candidate_labels=labels)\n",
    "    return result['labels'][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "be1c902f",
   "metadata": {},
   "outputs": [],
   "source": [
    "layoffs_df['reason_category'] = layoffs_df['reason_stated'].apply(classify_reason)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "cac00b30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>reason_stated</th>\n",
       "      <th>reason_category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Reduce bureaucracy and management layers</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>AI tools replace roles enabling smaller teams</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Pivot from metaverse to AI research</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Pivot to AI-first company strategy</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>AI data centres replace human ops</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>AI-driven efficiency and restructuring</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Post-Ansys acquisition restructuring</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>AI-forward strategy in customer ops</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>AI-forward content and marketing strategy</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Declining 5G demand and cost reduction</td>\n",
       "      <td>Cost Cutting</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Chip market slowdown</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Cost restructuring and portfolio focus</td>\n",
       "      <td>Cost Cutting</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>AI automation in warehouse operations</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>AI replaces QA and testing teams</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>Restructuring after Splunk acquisition</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>Restructuring sales teams</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>Slowing EV demand</td>\n",
       "      <td>Other</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>Store closures and cost restructuring</td>\n",
       "      <td>Cost Cutting</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>Cost restructuring with compensation</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>Internal reorganization</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>Post-CyberArk acquisition overlap</td>\n",
       "      <td>Acquisition</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>AI design tools replace human designers</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>Rethinking digital transformation</td>\n",
       "      <td>Restructuring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>AI investment and office space reduction</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>AI automates insurance tasks</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>Blast furnace closure</td>\n",
       "      <td>Cost Cutting</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>Offset AI infrastructure costs</td>\n",
       "      <td>AI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>Blast furnace closure effective</td>\n",
       "      <td>Cost Cutting</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    reason_stated reason_category\n",
       "0        Reduce bureaucracy and management layers   Restructuring\n",
       "1   AI tools replace roles enabling smaller teams              AI\n",
       "2             Pivot from metaverse to AI research              AI\n",
       "3              Pivot to AI-first company strategy              AI\n",
       "4               AI data centres replace human ops              AI\n",
       "5          AI-driven efficiency and restructuring              AI\n",
       "6            Post-Ansys acquisition restructuring   Restructuring\n",
       "7             AI-forward strategy in customer ops              AI\n",
       "8       AI-forward content and marketing strategy              AI\n",
       "9          Declining 5G demand and cost reduction    Cost Cutting\n",
       "10                           Chip market slowdown   Restructuring\n",
       "11         Cost restructuring and portfolio focus    Cost Cutting\n",
       "12          AI automation in warehouse operations              AI\n",
       "13               AI replaces QA and testing teams              AI\n",
       "14         Restructuring after Splunk acquisition   Restructuring\n",
       "15                      Restructuring sales teams   Restructuring\n",
       "16                              Slowing EV demand           Other\n",
       "17          Store closures and cost restructuring    Cost Cutting\n",
       "18           Cost restructuring with compensation   Restructuring\n",
       "19                        Internal reorganization   Restructuring\n",
       "20              Post-CyberArk acquisition overlap     Acquisition\n",
       "21        AI design tools replace human designers              AI\n",
       "22              Rethinking digital transformation   Restructuring\n",
       "23       AI investment and office space reduction              AI\n",
       "24                   AI automates insurance tasks              AI\n",
       "25                          Blast furnace closure    Cost Cutting\n",
       "26                 Offset AI infrastructure costs              AI\n",
       "27                Blast furnace closure effective    Cost Cutting"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "layoffs_df[['reason_stated','reason_category']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c3df009e",
   "metadata": {},
   "outputs": [],
   "source": [
    "layoffs_df.to_csv(\"layoffs_with_category.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4960e991",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "company    27\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "layoffs_df[['company']].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1d524b6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
