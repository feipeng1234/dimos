import React from 'react';
import {
  Alert,
  SafeAreaView,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import FigletText from '../utils/FigletText';

interface HomeScreenProps {
  onNavigateToSettings: () => void;
}

const HomeScreen: React.FC<HomeScreenProps> = ({onNavigateToSettings}) => {
  const handleAddRobotDog = () => {
    Alert.alert('Add Robot', 'This feature will be implemented soon!');
  };

  return (
    <SafeAreaView className="flex-1 bg-dimos-blue">
      <StatusBar barStyle="light-content" />
      <View className="pt-6 px-6 flex-row items-center justify-between">
        <FigletText text="DIMENSIONAL" color="#FFF200" fontSize={4} />
        <TouchableOpacity
          className="bg-dimos-yellow px-2 py-1 rounded-lg items-center justify-center"
          onPress={onNavigateToSettings}
          activeOpacity={0.7}
        >
          <View className="items-center justify-center">
            <Text className="text-2xl text-dimos-blue font-bold">⋯</Text>
          </View>
        </TouchableOpacity>
      </View>
      <View className="flex-1 items-center justify-center px-6">
        <View className="items-center justify-center -mt-10">
          <TouchableOpacity
            className="bg-dimos-yellow px-5 py-3 rounded-2xl items-center justify-center"
            activeOpacity={0.9}
            onPress={handleAddRobotDog}>
            <FigletText text="ADD ROBOT" color="#0016B1" fontSize={5}/>
          </TouchableOpacity>
        </View>
      </View>
      <View className="items-center pb-8 pt-4">
        <Text className="text-dimos-yellow text-xs font-mono">v0.0.1</Text>
      </View>
    </SafeAreaView>
  );
};

export default HomeScreen;
