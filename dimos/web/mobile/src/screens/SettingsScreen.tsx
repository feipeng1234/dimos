import React from 'react';
import {
  SafeAreaView,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  Alert,
} from 'react-native';
import FigletText from '../utils/FigletText';

interface SettingsScreenProps {
  onBack: () => void;
}

const SettingsScreen: React.FC<SettingsScreenProps> = ({onBack}) => {
  const handleAbout = () => {
    Alert.alert(
      'About',
      'Powering generalist robotics',
      [{text: 'OK'}]
    );
  };

  const handleTermsAndConditions = () => {
    Alert.alert(
      'Terms & Conditions',
      'By using this app, you agree to our terms of service. This app is provided as-is for educational and entertainment purposes.',
      [{text: 'OK'}]
    );
  };

  return (
    <SafeAreaView className="flex-1 bg-dimos-blue">
      <StatusBar barStyle="light-content" />

      <View className="flex-row items-center justify-between pt-6 px-6 pb-4">
        <TouchableOpacity
          className="bg-dimos-yellow px-1.5 py-1 rounded-lg items-center justify-center"
          onPress={onBack}
          activeOpacity={0.7}
        >
          <FigletText text="BACK" color="#0016B1" fontSize={2} />
        </TouchableOpacity>
        <FigletText text="SETTINGS" color="#FFF200" fontSize={4} />
        <View className="w-9" />
      </View>

      <ScrollView className="flex-1 px-6" showsVerticalScrollIndicator={false}>
        <View className="mt-6">
          <TouchableOpacity
            className="rounded-xl mb-3 border"
            style={{
              backgroundColor: 'rgba(255, 242, 0, 0.1)',
              borderColor: 'rgba(255, 242, 0, 0.2)',
            }}
            onPress={handleAbout}
            activeOpacity={0.7}
          >
            <View className="flex-row items-center justify-between px-3 py-3">
              <FigletText text="ABOUT" color="#FFF200" fontSize={2} />
              <FigletText text=">" color="#FFF200" fontSize={3} />
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            className="rounded-xl mb-3 border"
            style={{
              backgroundColor: 'rgba(255, 242, 0, 0.1)',
              borderColor: 'rgba(255, 242, 0, 0.2)',
            }}
            onPress={handleTermsAndConditions}
            activeOpacity={0.7}
          >
            <View className="flex-row items-center justify-between px-3 py-3">
              <FigletText text="TERMS & CONDITIONS" color="#FFF200" fontSize={2} />
              <FigletText text=">" color="#FFF200" fontSize={3} />
            </View>
          </TouchableOpacity>
        </View>
      </ScrollView>

      <View className="items-center pb-4 pt-6">
        <Text className="text-dimos-yellow text-xs opacity-70 font-mono">
          v0.0.1
        </Text>
      </View>
    </SafeAreaView>
  );
};

export default SettingsScreen;
