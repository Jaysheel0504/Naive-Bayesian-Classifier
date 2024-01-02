#include <bits/stdc++.h>
using namespace std;


struct Instance
{
	vector<string> image;
	int classification;			//1 if face, 0 if non-face
};

//global variables

vector<Instance> training_dataset;								//training set consisting of all 451 instances
vector<Instance> trainingset_faces, trainingset_nonfaces;		//training set containing faces and nonfaces respectively
vector<Instance> testing_dataset;								//testing set consisting of all 150 instances
vector<int> predicted_labels;									    //vector containing the classification of all test instances
int count_faces, count_nonfaces;

vector<vector<float>> probability_hash_given_face(70, vector<float>(60,0));
vector<vector<float>> probability_hash_given_nonface(70, vector<float>(60,0));
vector<vector<float>> probability_blank_given_face(70, vector<float>(60,0));
vector<vector<float>> probability_blank_given_nonface(70, vector<float>(60,0));

void getData(string data, string datalabels, vector<Instance>& dataset, bool update);
void calculateAllProbabilities();
void classify();
void show_results();

int main(){
	getData("training_data/training_data_faces.txt", "training_data/training_data_labels.txt", training_dataset, true);
    // cout<<training_dataset.size();
    // cout<<training_dataset[1].classification;
    getData("testing_data/testing_data_faces.txt", "testing_data/testing_data_labels.txt", testing_dataset, false);
    
	calculateAllProbabilities();

	// for(int i=0 ; i<70 ; i++){
	// 	for(int j=0 ; j<5 ; j++){
	// 		cout<<probability_blank_given_nonface[i][j]<<endl;
	// 	}
	// 	cout<<endl;
	// }

	classify();
	


	// for(int i=0 ; i<5 ; i++){
	// 	cout<<"FOR PIXEL "<<i+1<<endl;
	// 	cout<<"Hash + face "<<probability_hash_given_face[35][35+i]<<endl;
	// 	cout<<"Blank + face "<<probability_blank_given_face[35][35+i]<<endl;
	// 	cout<<"Hash + nonface "<<probability_hash_given_nonface[35][35+i]<<endl;
	// 	cout<<"Blank + nonface "<<probability_blank_given_nonface[35][35+i]<<endl;
	// }

	show_results();
}

void classify()
{
	for(int i=0;i<testing_dataset.size();i++)
	{
		long double prob_face = 1.0;
		long double prob_non_face = 1.0;
		//cout<<"here"<<endl;
		float factor = 1.36541;
		for(int j=0;j<70;j++)
		{
			for(int k=0;k<60;k++)
			{
				if(testing_dataset[i].image[j][k] == '#')
				{
					prob_face = prob_face * probability_hash_given_face[j][k] * factor;
					prob_non_face = prob_non_face * probability_hash_given_nonface[j][k] * factor;
				}
				else
				{
					prob_face = prob_face * probability_blank_given_face[j][k] * factor;
					prob_non_face = prob_non_face * probability_blank_given_nonface[j][k] * factor;

				}

			}
		}
		// cout<<"here2"<<endl;
		float prior_face = (count_faces* 1.0 / training_dataset.size());
		float prior_nonface = (count_nonfaces* 1.0 / training_dataset.size());
		// cout<<"Priors "<<prior_face<<"  "<<prior_nonface<<endl;
		// cout<<"Likelihoods "<<prob_face<<"  "<<prob_non_face<<endl;
		prob_face = prob_face * prior_face;
		prob_non_face = prob_non_face* prior_nonface;

		if(prob_face >= prob_non_face)
		{
			predicted_labels.push_back(1);
		}
		else{
			predicted_labels.push_back(0);
		}
	}

}

void show_results()
{
	int tp = 0, fn = 0, fp = 0, tn = 0;
	for (int i = 0; i < testing_dataset.size(); i++)
	{
		if (testing_dataset[i].classification == 1)
		{
			if (predicted_labels[i] == 1)
			{
				tp++;
			}
			else
			{
				fn++;
			}
		}
		else
		{
			if (predicted_labels[i] == 1)
			{
				fp++;
			}
			else
			{
				tn++;
			}
		}
	}
	cout << "Accuracy : " << ((tp + tn) / float(testing_dataset.size())) * 100 << "%\n\n";
	cout << "Confusion Matrix\n\n";
	cout << "Actually Face, Predicted Face : " << tp << "\n";
	cout << "Actually Face, Predicted Non-Face : " << fn << "\n";
	cout << "Actually Non-Face, Predicted Face : " << fp << "\n";
	cout << "Actually Non-Face, Predicted Non-Face : " << tn << "\n";
	cout << "\nPrecision : " << (tp / float(tp + fp)) * 100 << "%\n";
	cout << "\nRecall : " << (tp / float(tp + fn)) * 100<< "%\n";
}

void calculateAllProbabilities(){
	int hash_face=0, hash_nonface=0, blank_face=0, blank_nonface=0;
	for(int i=0 ; i<training_dataset.size() ; i++){
		for(int j=0 ; j<70 ; j++){
			for(int k=0 ; k<60 ; k++){
				if(training_dataset[i].classification==1){//face
					if(training_dataset[i].image[j][k] == '#'){//hash+face
						probability_hash_given_face[j][k] += 1;
					}
					else{//blank+face
						probability_blank_given_face[j][k] += 1;
					} 
				}else{//nonface
					if(training_dataset[i].image[j][k] == '#'){//hash+nonface
						probability_hash_given_nonface[j][k] +=1;
					}
					else{//blank+nonface
						probability_blank_given_nonface[j][k] += 1;
					}
				}
			}
		}
	}

	// cout<<probability_hash_given_face[35][35]<<" "
	// 	<<	probability_blank_given_face[35][35] <<" "
	// 	<<	probability_hash_given_nonface[35][35] <<" "
	// 	<<	probability_blank_given_nonface[35][35] <<" ";
    float pseudo = 1.9;
	for(int j=0 ; j<70 ; j++){
		for(int k=0 ; k<60 ; k++){
			probability_hash_given_face[j][k] += pseudo*0.5;
			probability_blank_given_face[j][k] += pseudo*0.5;
			probability_hash_given_nonface[j][k] += pseudo*0.5;
			probability_blank_given_nonface[j][k] += pseudo*0.5;
			probability_hash_given_face[j][k] /= (1.0*count_faces + pseudo);
			probability_blank_given_face[j][k] /= (1.0*count_faces + pseudo);
			probability_hash_given_nonface[j][k] /= (1.0*count_nonfaces + pseudo);
			probability_blank_given_nonface[j][k] /= (1.0*count_nonfaces + pseudo);
		}
	}
}

void getData(string data, string datalabels, vector<Instance>& dataset, bool update){
    string temp;
    ifstream file(datalabels);
	vector<int> labels;				//this array contains the classifications of the training/testing set
	for (int i = 0; getline(file, temp); i++){
		labels.push_back(stoi(temp));
	}
	file.close();

    file.open(data);
	for (int i=0 ; i < labels.size() ; i++)
	{
        Instance temp_instance;
        temp_instance.classification = labels[i];
		if(update){
			if(labels[i]==1) count_faces++;
			else count_nonfaces++;
		}
        dataset.push_back(temp_instance);
	}
	int i = 0, j = 1;
	while (getline(file, temp))
	{
		if(temp.length() < 60)
		{
			int x = temp.length();

			while(x < 60)
			{
				temp.push_back(' ');
				x++;
			}
		}
		dataset[i].image.push_back(temp);		
		j++;
		if ((j - 1) % 70 == 0)
		{
			i++;
		}
	}
	file.close();
}

