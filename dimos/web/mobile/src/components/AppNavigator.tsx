import React, {useState, useEffect, useCallback} from 'react';
import LoadingScreen from '../screens/LoadingScreen';
import HomeScreen from '../screens/HomeScreen';
import SettingsScreen from '../screens/SettingsScreen';

type Screen = 'loading' | 'home' | 'settings';

const AppNavigator: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<Screen>('loading');

  useEffect(() => {
    const timer = setTimeout(() => setCurrentScreen('home'), 1200);
    return () => clearTimeout(timer);
  }, []);

  const navigateToSettings = useCallback(() => {
    setCurrentScreen('settings');
  }, []);

  const navigateToHome = useCallback(() => {
    setCurrentScreen('home');
  }, []);

  if (currentScreen === 'loading') {
    return <LoadingScreen />;
  }

  if (currentScreen === 'settings') {
    return <SettingsScreen onBack={navigateToHome} />;
  }

  return <HomeScreen onNavigateToSettings={navigateToSettings} />;
};

export default AppNavigator;
